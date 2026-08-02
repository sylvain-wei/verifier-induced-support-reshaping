# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, top_k=0, student_top_k_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)

                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)

                                 # 4. If ulysses SP is active, the `logits_rmpad` /
                                 # `log_probs_all` tensors have already been padded
                                 # (`pad_size` zeros appended to dim 0 of input_ids_rmpad
                                 # before slicing) and then split across ranks — each
                                 # rank now holds `(total_nnz + pad_size) / sp_size`
                                 # rows. We must mirror that on topk_ids_rmpad, else
                                 # `log_probs_all.gather(dim=-1, index=topk_ids)` below
                                 # raises
                                 #   "Size does not match at dimension 0 expected
                                 #    index [total_nnz, K] to be no larger than self
                                 #    [(total_nnz+pad_size)/sp_size, V]".
                                 # Reproduced in
                                 # logs/20260422_143914_smoke_opd_topk_reverse_kl_k16.log
                                 # with sp_size=4 (index [6599, 16] vs self [1650, V]).
                                 # Pad with 0 (gather output on the padded slots is
                                 # discarded — those slots lie inside the pad_size tail
                                 # which response_mask masks out downstream).
                                 if self.use_ulysses_sp and pad_size > 0:
                                     topk_ids_rmpad = torch.nn.functional.pad(
                                         topk_ids_rmpad, (0, 0, 0, pad_size), value=0
                                     )
                                 if self.use_ulysses_sp:
                                     topk_ids_rmpad = slice_input_tensor(
                                         topk_ids_rmpad, dim=0, padding=False
                                     )

                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)

            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights

            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        # --- opd-lab RG-OPD gate hook -----------------------------------------
        # Idea 1 / F1: multiply rm_scores by a per-token reliability weight w_t
        # learned offline from the prefix-continuation oracle (Exp-2.2).
        #
        # Activated iff OPD_GATE_MODE in {"softgate"} and OPD_GATE_SIDECAR is
        # set to a stage7_reliability_predictor.json produced by
        # opd_lab/analysis/reliability_predictor.py.
        #
        # Features computed online on the student's own (B, T, K) top-K stream:
        #   * depth                 = t / response_length               (B, T)
        #   * tail_entropy_gap_mean = running-mean over last W tokens   (B, T)
        #   * tail_entropy_gap_std  = running-std  over last W tokens   (B, T)
        #   * [optional 4-core] tail_overlap_mass_mean                  (B, T)
        # The running windows are CAUSAL so the gate is computable during
        # online rollouts (no peek at the future).
        #
        # The gate:
        #   z_i = (feat_i - feat_mean_i) / feat_std_i       # sidecar scaler
        #   score = α_0 + Σ_i α_i · z_i
        #   w_t   = σ(score)                                # (B, T)
        #   rm_scores ← rm_scores · w_t.unsqueeze(-1)       # broadcast over K
        #
        # Reliability-stratified monitoring (per-step aggregate; written
        # next to the normal DA /_agg/ sidecars):
        #   * gate/w_mean, gate/w_std
        #   * gate/w_mean_bucket_top / mid / bot           (by w_t quintile)
        #   * gate/pg_like_bucket_top / mid / bot          (rm_score sumK)
        #
        # Schema for the sidecar JSON (produced by reliability_predictor.py):
        #   {
        #     "features": [...names...],
        #     "rg_opd_alpha_init": {
        #         "alpha_0": ..., "alpha_1_<feat0>": ..., ...
        #     },
        #     "model_on_full": {
        #         "intercept": ...,
        #         "coefficients": {"<feat_i>": ...},
        #         "feature_mean":  {"<feat_i>": ...},
        #         "feature_std":   {"<feat_i>": ...},
        #     }
        #   }
        _gate_mode = os.environ.get("OPD_GATE_MODE", "").strip()
        _gate_sidecar = os.environ.get("OPD_GATE_SIDECAR", "").strip()
        _gate_tail_window = int(os.environ.get("OPD_GATE_TAIL_WINDOW", "32"))
        _gate_weight_floor = float(os.environ.get("OPD_GATE_WEIGHT_FLOOR", "0.0"))
        _gate_norm = os.environ.get("OPD_GATE_NORM", "").strip().lower()
        # _gate_norm values:
        #   "" / "none"       → Form 2 (damping RG-OPD, current default)
        #   "batch_global"    → Form 1 (selection-normalized RG-OPD):
        #                       w_t ← w_t / w̄_batch  where w̄ = Σ(m·w)/Σ(m)
        #                       This makes masked_mean denominator effectively
        #                       weight-aware: Σ(m·(w/w̄)·ℓ)/Σ(m) = Σ(m·w·ℓ)/Σ(m·w)
        _gate_stats_out = {}  # per-step aggregates; picked up by DA agg sidecar

        if _gate_mode == "softgate" and _gate_sidecar and os.path.exists(_gate_sidecar):
            try:
                # Load once per dp_actor instance; cache on self.
                if not hasattr(self, "_rg_opd_gate_cache"):
                    import json as _json
                    with open(_gate_sidecar) as _fp:
                        _sidecar = _json.load(_fp)
                    _feats = list(_sidecar["features"])
                    _model = _sidecar["model_on_full"]
                    _intercept = float(_model["intercept"])
                    _coefs = [float(_model["coefficients"][f]) for f in _feats]
                    _mean = [float(_model["feature_mean"][f]) for f in _feats]
                    _std = [float(_model["feature_std"][f]) for f in _feats]
                    self._rg_opd_gate_cache = {
                        "features": _feats,
                        "intercept": _intercept,
                        "coefs": _coefs,
                        "mean": _mean,
                        "std": _std,
                    }
                    logger.info(
                        f"[opd-lab RG-OPD gate] loaded sidecar {_gate_sidecar}: "
                        f"features={_feats}, intercept={_intercept:.4f}, "
                        f"coefs={list(zip(_feats, _coefs))}"
                    )

                _g = self._rg_opd_gate_cache
                _feats = _g["features"]

                # We need _response_mask, _entropy_gap_bt, _overlap_mask here.
                # These are computed further down in DA-export for logging.
                # Compute them locally (cheap; we'll just reuse vars names).
                _response_mask_gate = data.batch.get("response_mask", None)
                if _response_mask_gate is None:
                    _attn = data.batch.get("attention_mask", None)
                    _responses = data.batch.get("responses", None)
                    if _attn is not None and _responses is not None:
                        _T_resp = _responses.shape[1]
                        _response_mask_gate = _attn[:, -_T_resp:].to(device)
                if _response_mask_gate is not None:
                    _response_mask_gate = _response_mask_gate.to(device).float()

                _B_gate, _T_gate, _K_gate = S_logp.shape

                # Top-K renormalized entropy per position (safe for -inf)
                def _renorm_entropy_gate(lp):
                    _finite = torch.isfinite(lp)
                    _lp_safe = torch.where(_finite, lp, torch.full_like(lp, -1e30))
                    _lse = torch.logsumexp(_lp_safe, dim=-1, keepdim=True)
                    _lp_norm = lp - _lse
                    _p = torch.where(_finite, _lp_norm.exp(), torch.zeros_like(lp))
                    _lp_n = torch.where(_finite, _lp_norm, torch.zeros_like(_lp_norm))
                    _ent = -(_p * _lp_n).sum(dim=-1)
                    return torch.nan_to_num(_ent, nan=0.0, posinf=0.0, neginf=0.0)

                _stu_H = _renorm_entropy_gate(S_logp)              # (B, T)
                _tch_H = _renorm_entropy_gate(T_on_S)              # (B, T)
                # NB: training-time entropy_gap keeps its SIGN (stu - tch),
                # matching the sign convention the offline predictor fit on the
                # stage6b parquet. The DA-export block above uses abs() for a
                # separate metric; we deliberately do NOT take abs() here.
                _entropy_gap_signed = (_stu_H - _tch_H)             # (B, T)

                # Causal running window stats (B, T) over last W positions.
                # We use cumulative-sum trick: O(B*T) memory, no python loop.
                _W = int(min(_gate_tail_window, _T_gate))

                def _running_window_mean(x_bt):
                    # returns (B, T) where out[b, t] = mean(x_bt[b, max(0,t-W+1):t+1])
                    _cs = torch.cumsum(x_bt, dim=1)  # (B, T)
                    _cs_shift = torch.nn.functional.pad(_cs, (_W, 0))[:, :_cs.shape[1]]
                    _sum = _cs - _cs_shift                               # window sum
                    _counts = torch.arange(1, _cs.shape[1] + 1, device=_cs.device
                                           ).clamp(max=_W).float().unsqueeze(0)
                    return _sum / _counts

                def _running_window_std(x_bt):
                    _m = _running_window_mean(x_bt)
                    _m2 = _running_window_mean(x_bt * x_bt)
                    _var = (_m2 - _m * _m).clamp(min=1e-12)
                    return _var.sqrt()

                _tail_mean = _running_window_mean(_entropy_gap_signed)   # (B, T)
                _tail_std  = _running_window_std(_entropy_gap_signed)    # (B, T)

                # Optional: tail_overlap_mass_mean (4-core). For only_stu
                # strategy, overlap mass per token = sum of p̃(teacher_of_student)
                # on the student top-K slots that ARE in teacher top-K.
                # A cheap proxy (and what the oracle extractor used) is
                # the exp(T_on_S) mass on the student top-K slots summed.
                _tail_ovm_mean = None
                if "tail_overlap_mass_mean" in _feats:
                    _ovm_bt = torch.exp(T_on_S).sum(dim=-1).clamp(max=1.0)  # (B, T)
                    _tail_ovm_mean = _running_window_mean(_ovm_bt)

                # Depth feature: t / response_length (both 0-indexed → (t+1)/L).
                if _response_mask_gate is not None:
                    _L = _response_mask_gate.sum(dim=1).clamp(min=1.0)  # (B,)
                    _idx = torch.arange(_T_gate, device=device).float().unsqueeze(0) + 1.0
                    _depth_bt = (_idx / _L.unsqueeze(1)).clamp(min=0.0, max=1.0)
                    _depth_bt = _depth_bt * _response_mask_gate + (1 - _response_mask_gate) * 0.5
                else:
                    _depth_bt = torch.full((_B_gate, _T_gate), 0.5, device=device)

                # Build feature tensor in the order specified by the sidecar
                _feature_name_to_tensor = {
                    "depth": _depth_bt,
                    "tail_entropy_gap_mean": _tail_mean,
                    "tail_entropy_gap_std":  _tail_std,
                }
                if _tail_ovm_mean is not None:
                    _feature_name_to_tensor["tail_overlap_mass_mean"] = _tail_ovm_mean

                # Any missing feature in sidecar → fall back to its mean (→ z=0)
                _score = torch.full((_B_gate, _T_gate), _g["intercept"], device=device)
                for _i, _fname in enumerate(_feats):
                    _m_i, _s_i, _a_i = _g["mean"][_i], _g["std"][_i], _g["coefs"][_i]
                    if _fname in _feature_name_to_tensor:
                        _z = (_feature_name_to_tensor[_fname] - _m_i) / max(_s_i, 1e-8)
                    else:
                        logger.warning(
                            f"[RG-OPD gate] feature {_fname} not computable online; "
                            f"using z=0 (effectively dropping this feature)"
                        )
                        _z = torch.zeros((_B_gate, _T_gate), device=device)
                    _score = _score + _a_i * _z

                _w_bt = torch.sigmoid(_score)  # (B, T)
                if _gate_weight_floor > 0:
                    _w_bt = _gate_weight_floor + (1.0 - _gate_weight_floor) * _w_bt

                # --- Selection normalization (Form 1 RG-OPD) ----------------
                # Transforms damping RG-OPD (Form 2) into selection-normalized
                # RG-OPD (Form 1) by dividing w_t by its batch-global masked
                # mean.  After normalization, w̄ ≈ 1 so the downstream
                # `masked_mean(loss, mask)` denominator is effectively weight-
                # aware:  Σ(m·(w/w̄)·ℓ)/Σ(m) = Σ(m·w·ℓ)/Σ(m·w).
                # Batch-global (not per-sequence) preserves cross-sequence
                # selection: a reliable sequence contributes more total gradient
                # than an unreliable one.
                _w_mean_global_raw = None  # track for monitoring
                if _gate_norm == "batch_global":
                    if _response_mask_gate is not None:
                        _local_wsum = (_w_bt * _response_mask_gate).sum()
                        _local_msum = _response_mask_gate.sum()
                    else:
                        _local_wsum = _w_bt.sum()
                        _local_msum = torch.tensor(
                            float(_w_bt.numel()), device=device
                        )
                    # All-reduce across FSDP ranks so every GPU divides
                    # by the same true batch-global w̄.
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(_local_wsum)
                        torch.distributed.all_reduce(_local_msum)
                    _w_mean_global_raw = _local_wsum / (_local_msum + 1e-8)
                    _w_bt = _w_bt / _w_mean_global_raw.detach().clamp(min=1e-4)

                # Apply weight, broadcasting over K
                rm_scores = rm_scores * _w_bt.unsqueeze(-1)

                # --- Reliability-stratified monitoring -----------------------
                # Mean w_t, and rm_score_sumK per quintile of w_t (top/mid/bot).
                if _response_mask_gate is not None:
                    _rm = _response_mask_gate.bool()
                    _w_flat = _w_bt[_rm]
                    _rms_flat = rm_scores.sum(dim=-1)[_rm]
                    if _w_flat.numel() > 10:
                        _q20, _q80 = torch.quantile(_w_flat, torch.tensor([0.2, 0.8], device=device))
                        _m_bot = _w_flat <= _q20
                        _m_top = _w_flat >= _q80
                        _m_mid = (~_m_bot) & (~_m_top)
                        def _mean_or_zero(t, m):
                            return float(t[m].mean().item()) if m.any() else 0.0
                        _gate_stats_out = {
                            "gate/w_mean": float(_w_flat.mean().item()),
                            "gate/w_std":  float(_w_flat.std().item()),
                            "gate/w_p20":  float(_q20.item()),
                            "gate/w_p80":  float(_q80.item()),
                            "gate/w_mean_bot20": _mean_or_zero(_w_flat, _m_bot),
                            "gate/w_mean_mid60": _mean_or_zero(_w_flat, _m_mid),
                            "gate/w_mean_top20": _mean_or_zero(_w_flat, _m_top),
                            "gate/rm_sumK_bot20": _mean_or_zero(_rms_flat, _m_bot),
                            "gate/rm_sumK_mid60": _mean_or_zero(_rms_flat, _m_mid),
                            "gate/rm_sumK_top20": _mean_or_zero(_rms_flat, _m_top),
                            "gate/weight_floor":  float(_gate_weight_floor),
                            "gate/tail_window":   int(_W),
                            "gate/n_features":    int(len(_feats)),
                            "gate/norm_mode":     1.0 if _gate_norm == "batch_global" else 0.0,
                        }
                        # For batch_global norm: log the raw w̄ (before norm)
                        # and the post-normalization mean (should ≈ 1.0).
                        if _gate_norm == "batch_global" and _w_mean_global_raw is not None:
                            _gate_stats_out["gate/w_mean_raw"] = float(_w_mean_global_raw.item())
                            _gate_stats_out["gate/w_mean_post_norm"] = float(_w_flat.mean().item())

                        # --- Depth-binned gate stats (Figure 5 diagnostic) ---
                        # Added 2026-04-26 evening after F1-success step ~88
                        # review: response_length rising + grad_norm ~0.5x
                        # D1B2 suggests gate may be over-suppressing late-
                        # depth tokens. We bin by depth ∈ [0,1] into 5 bins
                        # and log w_mean / rm_sumK / frac per bin.
                        try:
                            _depth_flat = _depth_bt[_rm]
                            # 5 bins: [0,0.2), [0.2,0.4), ..., [0.8,1.0]
                            _bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-6]
                            _bin_labels = ["d00_20", "d20_40", "d40_60", "d60_80", "d80_100"]
                            _total_tokens = _w_flat.numel()
                            for _bi in range(5):
                                _m_bin = (_depth_flat >= _bin_edges[_bi]) & (_depth_flat < _bin_edges[_bi + 1])
                                _lbl = _bin_labels[_bi]
                                _n_bin = int(_m_bin.sum().item())
                                _gate_stats_out[f"gate/frac_{_lbl}"] = _n_bin / max(_total_tokens, 1)
                                if _n_bin > 0:
                                    _gate_stats_out[f"gate/w_mean_{_lbl}"] = _mean_or_zero(_w_flat, _m_bin)
                                    _gate_stats_out[f"gate/rm_sumK_{_lbl}"] = _mean_or_zero(_rms_flat, _m_bin)
                                else:
                                    _gate_stats_out[f"gate/w_mean_{_lbl}"] = 0.0
                                    _gate_stats_out[f"gate/rm_sumK_{_lbl}"] = 0.0
                        except Exception as _de:
                            logger.warning(f"[RG-OPD gate] depth-bin aggregation failed (non-fatal): {_de}")
                else:
                    _gate_stats_out = {
                        "gate/w_mean": float(_w_bt.mean().item()),
                        "gate/norm_mode": 1.0 if _gate_norm == "batch_global" else 0.0,
                    }
            except Exception as _e:
                logger.warning(f"[RG-OPD gate] failed, falling back to ungated rm_scores: {_e}")
                _gate_stats_out = {"gate/error": 1.0}
        # --- end RG-OPD gate hook --------------------------------------------

        # --- opd-lab DA (Data-Analysis) export hook --------------------------
        # Two-tier export for Phase E / Idea 1:
        #   (a) per-step aggregate scalars  → JSON sidecar at every step,
        #       picked up by ray_trainer and merged into metrics (→ WandB).
        #   (b) per-token parquet shards    → every OPD_TOKEN_STATS_EVERY steps
        #       for offline reliability-predictor training.
        # Rank-0 guard: only the first FSDP rank writes (all ranks hold the
        # same driver-dispatched `rm_scores`; a minor simplification that's
        # fine for DA — offline analysis re-aggregates per step anyway).
        _stats_dir = os.environ.get("OPD_TOKEN_STATS_DIR", "")
        _global_step = int(data.meta_info.get("global_steps", 0) or 0)
        _export_every = int(os.environ.get("OPD_TOKEN_STATS_EVERY", "50") or "50")
        _agg_every = int(os.environ.get("OPD_TOKEN_STATS_AGG_EVERY", "1") or "1")
        try:
            _is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        except Exception:
            _is_rank0 = True
        if _stats_dir and _is_rank0 and _global_step > 0:
            _want_parquet = (_global_step % _export_every == 0)
            _want_agg = (_global_step % _agg_every == 0)
            if _want_parquet or _want_agg:
                try:
                    _response_mask = data.batch.get("response_mask", None)
                    if _response_mask is None:
                        _attn = data.batch.get("attention_mask", None)
                        _responses = data.batch.get("responses", None)
                        if _attn is not None and _responses is not None:
                            _T = _responses.shape[1]
                            _response_mask = _attn[:, -_T:].to(device)
                    if _response_mask is not None:
                        _response_mask = _response_mask.to(device)

                    _B, _T_k, _K = S_logp.shape

                    # Per-token overlap ratio: for only_stu, overlap_mask is
                    # (B, T, K) where entry==1 iff the student's k-th top id
                    # is ALSO in the teacher's top-K. Mean over K → (B, T).
                    _overlap_ratio_bt = None
                    if overlap_mask is not None and overlap_mask.dim() == 3:
                        _overlap_ratio_bt = overlap_mask.float().mean(dim=-1)

                    # --- Strategy-aware "active support" selection ----------
                    # We report H(p), H(q) and the top-0 log-probs on the SAME
                    # top-K support set that the loss objective actually uses,
                    # so the entropy gap Δ_H stays a consistent comparison of
                    # renormalized p and q on a shared support.
                    #
                    # Before 2026-04-24 this block always pulled teacher stats
                    # from `T_on_S = teacher_on_student_log_probs`, which is
                    # the right semantic for strategy == "only_stu" but NOT for
                    # "only_tch" / "intersection": in those branches
                    # `_compute_teacher_top_k_log_probs` returns an intersection
                    # lookup with -inf fills outside the intersection, and the
                    # logsumexp blew up to -inf → teacher_entropy_topk=NaN,
                    # teacher_logp_top0=-inf in the WandB run. See
                    # exp_debug_logs/20260424_DA_export_only_tch_nan.md for the
                    # full diagnosis and the B3 success/failure run evidence.
                    if strategy == "only_stu":
                        _S_lp_supp = S_logp               # (B, T, K_s)
                        _T_lp_supp = T_on_S               # (B, T, K_s)
                    elif strategy == "only_tch":
                        # loss uses teacher top-K support; p̃ from S_on_T, q̃ from T_logp
                        _S_lp_supp = S_on_T if S_on_T is not None else S_logp
                        _T_lp_supp = T_logp if T_logp is not None else T_on_S
                    elif strategy == "intersection":
                        # keep student top-K geometry but hard-mask to intersection
                        if overlap_mask is not None:
                            _im = overlap_mask.bool()
                            _neg = torch.full_like(S_logp, float("-inf"))
                            _S_lp_supp = torch.where(_im, S_logp, _neg)
                            _T_lp_supp = torch.where(_im, T_on_S, _neg)
                        else:
                            _S_lp_supp = S_logp
                            _T_lp_supp = T_on_S
                    elif strategy in ("union", "union-intersection"):
                        # loss body concatenates S top-K with S-on-T top-K (for q analogously)
                        _S_lp_supp = torch.cat(
                            [S_logp, S_on_T] if S_on_T is not None else [S_logp], dim=-1
                        )
                        _T_lp_supp = torch.cat(
                            [T_on_S, T_logp] if T_logp is not None else [T_on_S], dim=-1
                        )
                    else:
                        _S_lp_supp = S_logp
                        _T_lp_supp = T_on_S

                    def _renorm_entropy(lp):
                        # Top-K renormalized entropy, -inf-safe.
                        # If a token position has every entry = -inf (empty
                        # intersection), we return 0 entropy instead of NaN.
                        _finite = torch.isfinite(lp)
                        _lp_for_lse = torch.where(
                            _finite, lp, torch.full_like(lp, -1e30)
                        )
                        _lse = torch.logsumexp(_lp_for_lse, dim=-1, keepdim=True)
                        _lp_norm = lp - _lse
                        _p_norm = torch.where(
                            _finite, _lp_norm.exp(), torch.zeros_like(lp)
                        )
                        _lp_norm_safe = torch.where(
                            _finite, _lp_norm, torch.zeros_like(_lp_norm)
                        )
                        _ent = -(_p_norm * _lp_norm_safe).sum(dim=-1)
                        return torch.nan_to_num(_ent, nan=0.0, posinf=0.0, neginf=0.0)

                    _student_entropy_bt = _renorm_entropy(_S_lp_supp)
                    _teacher_entropy_bt = _renorm_entropy(_T_lp_supp)

                    _entropy_gap_bt = (_student_entropy_bt - _teacher_entropy_bt).abs()

                    # Top-0 log-probs: log-prob at the support slot with the
                    # HIGHEST probability under the distribution that defines
                    # the support. For only_stu that's the student's argmax
                    # (so teacher_logp_top0 = q(argmax p)); for only_tch it's
                    # the teacher's argmax (so student_logp_top0 = p(argmax q)).
                    _student_logp_top0_bt = torch.nan_to_num(
                        _S_lp_supp[:, :, 0], nan=0.0, posinf=0.0, neginf=-80.0
                    )
                    _teacher_logp_top0_bt = torch.nan_to_num(
                        _T_lp_supp[:, :, 0], nan=0.0, posinf=0.0, neginf=-80.0
                    )

                    # Per-token distillation reward (signed): rm_scores sum
                    # over K, sign matches the advantage pipeline.
                    _token_losses_bt = rm_scores.sum(dim=-1)

                    if _want_agg:
                        # --- (a) per-step aggregate scalars → JSON sidecar ---
                        import json
                        _os_path = os.path.join(_stats_dir, "_agg")
                        os.makedirs(_os_path, exist_ok=True)
                        if _response_mask is not None:
                            _rm = _response_mask.float()
                            _denom = _rm.sum().clamp(min=1.0)
                            def _mm(t):
                                return ((t * _rm).sum() / _denom).item()
                            _agg = {
                                "da/student_entropy_topk": _mm(_student_entropy_bt),
                                "da/teacher_entropy_topk": _mm(_teacher_entropy_bt),
                                "da/entropy_gap_topk": _mm(_entropy_gap_bt),
                                "da/student_logp_top0": _mm(_student_logp_top0_bt),
                                "da/teacher_logp_top0": _mm(_teacher_logp_top0_bt),
                                "da/rm_score_sumK": _mm(_token_losses_bt),
                                "da/response_tokens": float(_denom.item()),
                                "da/valid_batch_size": int(_B),
                            }
                            if _overlap_ratio_bt is not None:
                                _agg["da/overlap_ratio_topk"] = _mm(_overlap_ratio_bt)
                        else:
                            _agg = {}
                        _agg["da/global_step"] = int(_global_step)
                        _agg["da/topk_strategy"] = strategy
                        _agg["da/topk_k"] = int(top_k)
                        # Merge RG-OPD gate monitoring (if active)
                        if _gate_stats_out:
                            _agg.update(_gate_stats_out)
                        with open(os.path.join(_os_path, f"step_{_global_step:07d}.json"), "w") as _f:
                            json.dump(_agg, _f)

                    if _want_parquet:
                        # --- (b) per-token parquet shard ---
                        from opd_lab.analysis.export_token_stats import TokenStatsExporter
                        from opd_lab.algorithms.token_weighting import TokenStats

                        _sampled_ids = data.batch.get("responses", None)
                        if _sampled_ids is not None:
                            _sampled_ids = _sampled_ids.to(device)

                        if _response_mask is not None:
                            _resp_lens = _response_mask.sum(dim=1).long()
                        else:
                            _resp_lens = torch.full((_B,), _T_k, dtype=torch.long, device=device)

                        _stats = TokenStats(
                            overlap_ratio=_overlap_ratio_bt.detach().cpu() if _overlap_ratio_bt is not None else None,
                            entropy_gap=_entropy_gap_bt.detach().cpu(),
                            student_entropy=_student_entropy_bt.detach().cpu(),
                            teacher_entropy=_teacher_entropy_bt.detach().cpu(),
                            student_logp_sampled=_student_logp_top0_bt.detach().cpu(),
                            teacher_logp_sampled=_teacher_logp_top0_bt.detach().cpu(),
                            token_mask=_response_mask.detach().cpu() if _response_mask is not None else None,
                            response_lengths=_resp_lens.detach().cpu(),
                        )

                        _exporter = TokenStatsExporter(_stats_dir, export_every=1)
                        _exporter.export(
                            global_step=_global_step,
                            token_stats=_stats,
                            token_losses=_token_losses_bt.detach().cpu(),
                            sampled_token_ids=_sampled_ids.detach().cpu() if _sampled_ids is not None else None,
                            config_info={"topk_strategy": strategy, "topk_k": int(top_k)},
                        )
                except Exception as _e:
                    logger.warning(f"[opd-lab DA] token stats export failed at step {_global_step}: {_e}")

        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, _, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = log_prob

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
