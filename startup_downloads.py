"""
Startup background downloader for hf_models.toml.

On ComfyUI startup, reads /workspace/hf_models.toml and downloads missing
HuggingFace models sequentially in a background thread using hf_hub_download.
Files are symlinked into the models directory (same behaviour as hf_downloads.py).
"""

import os
import re
import threading
from pathlib import Path
from typing import Optional

TOML_PATH = "/workspace/hf_models.toml"
SKIP_ENV = "SKIP_MODEL_DOWNLOADS"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _should_skip() -> bool:
    return os.environ.get(SKIP_ENV, "").strip().lower() in TRUE_VALUES


def _parse_toml(path: str) -> Optional[dict]:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            print("[HF Downloader Startup] tomllib not available, cannot parse config")
            return None
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"[HF Downloader Startup] Failed to parse {path}: {e}")
        return None


def _category_enabled(category: str) -> bool:
    if not category:
        return True
    categories = [c.strip() for c in category.split(",") if c.strip()]
    if not categories:
        return True
    for cat in categories:
        env_var = "DOWNLOAD_" + re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_").upper()
        if os.environ.get(env_var, "").strip().lower() in TRUE_VALUES:
            return True
    return False


def _worker(items: list, models_dir: str) -> None:
    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        print("[HF Downloader Startup] huggingface_hub not available")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"[HF Downloader Startup] Warning: HF login failed: {e}")

    print(f"[HF Downloader Startup] Downloading {len(items)} HF models in background...")
    for item in items:
        target_path = Path(models_dir) / item["target_rel_path"]

        if target_path.exists():
            continue  # Already present (follows symlinks), silent skip

        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[HF Downloader Startup] Downloading: {item['target_rel_path']}")
            cached_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["repo_path"],
                token=hf_token,
                resume_download=True,
            )
            # Symlink from models dir into HF cache (same as hf_downloads.py)
            if target_path.is_symlink():
                target_path.unlink()
            target_path.symlink_to(cached_path)
            print(f"[HF Downloader Startup] Done: {item['target_rel_path']}")
        except Exception as e:
            print(f"[HF Downloader Startup] Failed: {item['target_rel_path']}: {e}")

    print("[HF Downloader Startup] All HF background downloads complete.")


def start_background_downloads() -> None:
    if _should_skip():
        print("[HF Downloader Startup] SKIP_MODEL_DOWNLOADS set, skipping.")
        return

    if not os.path.exists(TOML_PATH):
        print(f"[HF Downloader Startup] No config at {TOML_PATH}, skipping.")
        return

    data = _parse_toml(TOML_PATH)
    if not data:
        return

    all_items = data.get("model", [])
    if not isinstance(all_items, list):
        return

    items = []
    for entry in all_items:
        if not isinstance(entry, dict):
            continue
        repo_id = str(entry.get("repo_id", "")).strip()
        repo_path = str(entry.get("repo_path", "")).strip()
        target_rel_path = str(entry.get("target_rel_path", "")).strip()
        if not repo_id or not repo_path or not target_rel_path:
            continue
        category = str(entry.get("category", "")).strip()
        if not _category_enabled(category):
            continue
        items.append({
            "repo_id": repo_id,
            "repo_path": repo_path,
            "target_rel_path": target_rel_path,
        })

    if not items:
        print("[HF Downloader Startup] No HF models to download.")
        return

    try:
        import folder_paths
        models_dir = folder_paths.models_dir
    except Exception:
        print("[HF Downloader Startup] Could not get models_dir from folder_paths.")
        return

    thread = threading.Thread(
        target=_worker,
        args=(items, models_dir),
        daemon=True,
        name="hf-startup-downloads",
    )
    thread.start()
