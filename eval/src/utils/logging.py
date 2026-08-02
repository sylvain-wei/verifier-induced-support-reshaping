"""
Lightweight logging helpers. We deliberately keep this tiny and
stdlib-only so it works on any Python version. Logs go to both stdout
and `eval/logs/{run_id}/{module}.log`.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

DEFAULT_FMT = "[%(asctime)s] %(levelname).1s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_ROOT = str(Path(os.environ.get("PROJECT_ROOT", ".")) / "eval" / "logs")


def get_logger(
    name: str,
    run_id: Optional[str] = None,
    log_root: str = DEFAULT_LOG_ROOT,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(DEFAULT_FMT, DEFAULT_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if run_id:
        out_dir = Path(log_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / f"{name.replace('.', '_')}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
