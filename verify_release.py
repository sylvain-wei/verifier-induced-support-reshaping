#!/usr/bin/env python3
"""Run release-safety and integrity checks without third-party dependencies."""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IS_OPEN_SOURCE = ROOT.name == "open_source_package"
REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "REPRODUCIBILITY.md",
    "THIRD_PARTY_NOTICES.md",
    ".gitignore",
    ".gitattributes",
    ".github/workflows/release-check.yml",
    ".github/workflows/pages.yml",
    "MANIFEST.tsv",
    "scripts/rebuild_manifest.py",
    "configs/dapo/main_dapo.py",
    "eval/requirements-test.txt",
    "eval/scripts/run_rq1_required.sh",
    "scripts_eval/eval_one.py",
    "docs/index.html",
    "docs/styles.css",
    "docs/script.js",
    "docs/assets/figures/README.md",
)
JUNK_NAMES = {".DS_Store", "Thumbs.db", "tea_debug.log"}
JUNK_SUFFIXES = {".pyc", ".pyo", ".swp"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
VISUAL_TOKENS = (
    "matplotlib",
    "seaborn",
    "plotly",
    "cairosvg",
    "savefig(",
    "paper_plot_style",
)
MAX_GIT_BLOB_BYTES = 95 * 1024 * 1024
DOC_IMAGE_ALLOWLIST = frozenset(
    {
        "docs/assets/favicon.svg",
        "docs/assets/og-card.png",
        "docs/assets/figures/fig1-overview.svg",
        "docs/assets/figures/fig2-if-polarization.svg",
        "docs/assets/figures/fig3-math-searchability.svg",
        "docs/assets/figures/fig6-opening-divergence.svg",
        "docs/assets/figures/fig8-opening-intervention-a.svg",
        "docs/assets/figures/fig8-opening-intervention-b.svg",
        "docs/assets/figures/fig9-dri-prior.svg",
        "docs/assets/figures/fig11-teacher-state.svg",
    }
)


def first_party_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and "verl" not in path.relative_to(ROOT).parts
    ]


def all_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]


def check_manifest(errors: list[str]) -> None:
    manifest = ROOT / "MANIFEST.tsv"
    if not manifest.is_file():
        errors.append("MANIFEST.tsv is missing")
        return
    expected_files = {
        path.relative_to(ROOT).as_posix()
        for path in all_files()
        if path != manifest
    }
    listed: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 3 or fields[1] != "CLEANED" or not re.fullmatch(r"[0-9a-f]{64}", fields[2]):
            errors.append(f"malformed manifest row {line_number}")
            continue
        if fields[0] in listed:
            errors.append(f"duplicate manifest path: {fields[0]}")
        listed[fields[0]] = fields[2]
    if set(listed) != expected_files:
        errors.append("manifest coverage does not match package files")
    for rel_path, expected_hash in listed.items():
        file_path = ROOT / rel_path
        if not file_path.is_file():
            continue
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"manifest hash mismatch: {rel_path}")


def check_python(errors: list[str], files: list[Path]) -> None:
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 10),
            )
        except Exception as exc:
            errors.append(f"python syntax error: {path.relative_to(ROOT)}: {exc}")
            continue
        imports_os = any(
            isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names)
            for node in ast.walk(tree)
        )
        uses_os = any(
            isinstance(node, ast.Name) and node.id == "os" and isinstance(node.ctx, ast.Load)
            for node in ast.walk(tree)
        )
        if uses_os and not imports_os:
            errors.append(f"missing import os: {path.relative_to(ROOT)}")


def check_sensitive_content(errors: list[str], files: list[Path]) -> None:
    private_path = re.compile(r"/(?:Users|apdcephfs)/|/home/(?!dpsk_a2a(?:/|$))[A-Za-z0-9._-]+/")
    secret_patterns = (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}"),
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    )
    malformed_markers = (
        "fos." + "environ",
        "verl-" + "v0.4.1.x",
        '"os.' + 'environ.get("PROJECT_ROOT"',
    )
    for path in files:
        if path == Path(__file__).resolve() or path.name == "MANIFEST.tsv":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel_path = path.relative_to(ROOT)
        if private_path.search(text):
            errors.append(f"private absolute path marker: {rel_path}")
        if any(marker in text for marker in malformed_markers):
            errors.append(f"stale path-sanitization marker: {rel_path}")
        if any(pattern.search(text) for pattern in secret_patterns):
            errors.append(f"secret-like content: {rel_path}")


def check_open_source_scope(errors: list[str], files: list[Path]) -> None:
    if not IS_OPEN_SOURCE:
        return
    for path in files:
        if path == Path(__file__).resolve():
            continue
        rel_path = path.relative_to(ROOT)
        if path.suffix.lower() in IMAGE_SUFFIXES and rel_path.as_posix() not in DOC_IMAGE_ALLOWLIST:
            errors.append(f"first-party image asset in open-source package: {rel_path}")
        if path.suffix == ".py":
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in VISUAL_TOKENS):
                errors.append(f"direct visualization code in open-source package: {rel_path}")


def check_public_metadata(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    title = "Verifier-Induced Support Reshaping in On-Policy Optimization"
    if "anonymous open-source" in readme.lower():
        errors.append("README still describes an anonymous release")
    if title not in readme or title not in citation:
        errors.append("public title is inconsistent across README and CITATION.cff")
    for public_url in (
        "https://arxiv.org/abs/2608.00220",
        "https://sylvain-wei.github.io/verifier-induced-support-reshaping/",
        "https://github.com/sylvain-wei/verifier-induced-support-reshaping",
    ):
        if public_url not in readme or public_url not in citation:
            errors.append(f"public metadata URL is inconsistent: {public_url}")
    for required_notice in ("volcengine/verl", "google-research", "allenai/IFBench"):
        if required_notice not in notices:
            errors.append(f"third-party notice missing: {required_notice}")


def check_git_tracking(errors: list[str], files: list[Path]) -> None:
    """Reject release files that a fresh `git add .` would silently ignore."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    git_root = Path(result.stdout.strip())
    candidates = [path.relative_to(git_root).as_posix() for path in files]
    check = subprocess.run(
        ["git", "-C", str(git_root), "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(candidates) + "\0").encode(),
        capture_output=True,
    )
    if check.returncode not in (0, 1):
        errors.append("git ignore audit could not be completed")
        return
    for ignored in check.stdout.decode(errors="replace").split("\0"):
        if ignored:
            rel_path = (git_root / ignored).resolve().relative_to(ROOT)
            errors.append(f"release file ignored by Git: {rel_path}")


def main() -> int:
    errors: list[str] = []
    for rel_path in REQUIRED:
        if not (ROOT / rel_path).is_file():
            errors.append(f"required file missing: {rel_path}")
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {path.relative_to(ROOT)}")
        if path.is_file() and (path.name in JUNK_NAMES or path.suffix.lower() in JUNK_SUFFIXES):
            errors.append(f"junk file: {path.relative_to(ROOT)}")
        if path.is_file() and path.stat().st_size > MAX_GIT_BLOB_BYTES:
            errors.append(f"Git blob exceeds 95 MiB: {path.relative_to(ROOT)}")
    files = first_party_files()
    check_python(errors, files)
    released_files = all_files()
    check_sensitive_content(errors, released_files)
    check_open_source_scope(errors, files)
    check_public_metadata(errors)
    check_git_tracking(errors, released_files)
    check_manifest(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAILED: {len(errors)} release check(s)")
        return 1
    print(f"OK: {ROOT.name} passed release checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
