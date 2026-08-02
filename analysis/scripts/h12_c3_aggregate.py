#!/usr/bin/env python
"""H12 C.3 position sweep aggregator.

Reads all `*_pos{N}__shard*.parquet` files (plus the corresponding pos1 baselines
already in `f1_intervention/`) and produces:
  - analysis/tables/h12_c3_position_sweep.csv  (per-(strategy, primary, position))
  - analysis/tables/h12_c3_pos1_baselines.csv  (pos1 reference for the 4 strategies)

Metrics per (strategy, primary, position):
  best_at_32         per-prompt max(acc) averaged over 30 prompts
  avg_at_32          mean(acc) over all 960 rows
  dri_rate           fraction of rows with mode==DRI
  dai_rate           fraction of rows with mode==DAI
  other_rate         fraction of rows with mode==Other
  mean_resp_chars    mean response_chars
  n_rows             total rows

Effect sizes (relative to free baseline at pos1) are derived in the figures step.
"""
import argparse, glob, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('PROJECT_ROOT', '.'))
DATA = ROOT / 'analysis/data/f1_intervention'
OUT  = ROOT / 'analysis/tables'
OUT.mkdir(parents=True, exist_ok=True)

# strategy -> (pos1 condition name, primary)
POS1_MAP = {
    'single_token_C1':    ('single_token_C1',    'B_q3'),
    'single_token_C2':    ('single_token_C2',    'I_q3'),
    'random_top2_B_q3':   ('random_top2_B_q3',   'B_q3'),
    'random_top2_I_q3':   ('random_top2_I_q3',   'I_q3'),
}
# strategy -> primary (used for pos>1; primary is encoded in script & condition)
STRAT_PRIMARY = {
    'single_token_C1':    'B_q3',
    'single_token_C2':    'I_q3',
    'random_top2_B_q3':   'B_q3',
    'random_top2_I_q3':   'I_q3',
}
POSITIONS = [1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 50, 75, 100]


def load_condition(cond: str) -> pd.DataFrame:
    pat = str(DATA / f'{cond}__shard*.parquet')
    fs = sorted(glob.glob(pat))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def aggregate(df: pd.DataFrame, strategy: str, primary: str, position: int) -> dict:
    if df.empty:
        return None
    n_prompts = df['prompt_id'].nunique()
    best32 = df.groupby('prompt_id')['acc'].max().mean()
    avg32  = df['acc'].mean()
    n = len(df)
    mode_counts = df['mode'].value_counts(normalize=True)
    return {
        'strategy': strategy,
        'primary':  primary,
        'position': position,
        'n_rows':   n,
        'n_prompts': n_prompts,
        'best_at_32': best32,
        'avg_at_32':  avg32,
        'dri_rate':   mode_counts.get('DRI', 0.0),
        'dai_rate':   mode_counts.get('DAI', 0.0),
        'csi_rate':   mode_counts.get('CSI', 0.0),
        'other_rate': mode_counts.get('Other', 0.0),
        'mean_resp_chars': df['response_chars'].mean(),
    }


def main():
    rows = []

    # pos1 baselines (the existing 4 conditions)
    for strategy, (cond, primary) in POS1_MAP.items():
        df = load_condition(cond)
        r = aggregate(df, strategy, primary, 1)
        if r: rows.append(r)

    # pos > 1 (new sweep)
    for strategy, primary in STRAT_PRIMARY.items():
        for p in POSITIONS:
            if p == 1: continue
            cond = f'{strategy}_pos{p}'
            df = load_condition(cond)
            r = aggregate(df, strategy, primary, p)
            if r: rows.append(r)

    # Also include the 'free' baseline (no intervention) at pos1 as a reference
    df_free = load_condition('free')
    if not df_free.empty:
        rows.append({
            'strategy': 'free', 'primary': 'B_q3', 'position': 0,
            'n_rows': len(df_free), 'n_prompts': df_free['prompt_id'].nunique(),
            'best_at_32': df_free.groupby('prompt_id')['acc'].max().mean(),
            'avg_at_32':  df_free['acc'].mean(),
            'dri_rate':   (df_free['mode']=='DRI').mean(),
            'dai_rate':   (df_free['mode']=='DAI').mean(),
            'csi_rate':   (df_free['mode']=='CSI').mean(),
            'other_rate': (df_free['mode']=='Other').mean(),
            'mean_resp_chars': df_free['response_chars'].mean(),
        })

    out = pd.DataFrame(rows).sort_values(['strategy','position']).reset_index(drop=True)
    csv_path = OUT / 'h12_c3_position_sweep.csv'
    out.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}  ({len(out)} rows)')
    # Pretty print
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
