#!/usr/bin/env python
"""Tiny DeepSeek-V4-Pro chat client used by response-labeling utilities.

Reads `DEEPSEEK_API_KEY` from environment. Endpoint per DeepSeek public docs:
  https://api.deepseek.com/v1/chat/completions

Usage:
  from deepseek_client import DeepseekClient
  c = DeepseekClient(model="deepseek-v4-pro")
  out = c.chat([{"role":"user", "content": "hello"}])
"""
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")


class DeepseekClient:
    def __init__(self, model=DEFAULT_MODEL, base=DEFAULT_BASE, api_key=None,
                 max_retries=4, retry_backoff=2.0):
        self.model = model
        self.base = base.rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Export it before running."
            )
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def chat(self, messages, *, temperature=0.0, max_tokens=512,
             response_format=None, timeout=60):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                # OpenAI-style schema: choices[0].message.content
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as e:
                wait = self.retry_backoff ** attempt
                err = repr(e)[:200]
                print(f"[deepseek warn] attempt {attempt+1}/{self.max_retries}: {err}; wait {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"DeepSeek call failed after {self.max_retries} retries.")


def main():
    """CLI smoke test."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Reply OK in JSON: {\"ok\": true}")
    args = ap.parse_args()
    c = DeepseekClient()
    out = c.chat([{"role": "user", "content": args.prompt}], max_tokens=64)
    print(out)


if __name__ == "__main__":
    main()
