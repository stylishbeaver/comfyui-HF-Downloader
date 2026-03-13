"""
Startup background downloader for hf_models.toml.

On ComfyUI startup, reads /workspace/hf_models.toml and queues missing
HuggingFace models into the existing HF Downloader download pipeline via
asyncio.run_coroutine_threadsafe. The actual downloading uses exactly the
same mechanism as manual downloads from the UI (HFDownloader.download_and_merge).
"""

import asyncio
import os
import re
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


def _task_id(item: dict) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{item['repo_id']}_{item['repo_path']}")
    return f"hf_startup_{safe}"


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

    try:
        from .server_routes import download_tasks, download_progress, run_download_task
        import server
        loop = server.PromptServer.instance.loop
    except Exception as e:
        print(f"[HF Downloader Startup] Could not access download infrastructure: {e}")
        return

    queued = 0
    for item in items:
        target_path = Path(models_dir) / item["target_rel_path"]
        if target_path.exists():
            continue

        # Decompose repo_path into folder_path (dir within repo) + filename
        repo_path = item["repo_path"]
        if "/" in repo_path:
            folder_path, filename = repo_path.rsplit("/", 1)
        else:
            folder_path, filename = "root", repo_path

        # Decompose target_rel_path into output_dir + output_name (stem)
        target_rel = item["target_rel_path"]
        if "/" in target_rel:
            target_dir, target_filename = target_rel.rsplit("/", 1)
            output_dir = str(Path(models_dir) / target_dir)
        else:
            output_dir = str(models_dir)
            target_filename = target_rel
        output_name = Path(target_filename).stem

        task_id = _task_id(item)
        name = item["target_rel_path"]

        # Pre-register as queued so the Status tab shows the full list upfront
        download_progress[task_id] = {
            "status": "queued",
            "stage": "queued",
            "current": 0,
            "total": 0,
            "message": "Waiting to download...",
            "name": name,
            "is_startup": True,
        }

        # Schedule via the existing download pipeline — same path as UI downloads
        future = asyncio.run_coroutine_threadsafe(
            run_download_task(
                task_id=task_id,
                repo_id=item["repo_id"],
                model_path=folder_path,
                files=[filename],
                output_dir=output_dir,
                output_name=output_name,
                name=name,
            ),
            loop,
        )
        download_tasks[task_id] = future
        queued += 1
        print(f"[HF Downloader Startup] Queued: {name}")

    if queued:
        print(f"[HF Downloader Startup] {queued} model(s) queued for background download.")
    else:
        print("[HF Downloader Startup] All models already present, nothing to download.")
