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
STARTUP_CONCURRENCY_ENV = "HF_STARTUP_MAX_CONCURRENT"
PREREQ_STAGE = 5       # text encoders, VAEs — needed before anything can run
BASE_MODEL_STAGE = 10  # diffusion models
LORA_STAGE = 40
HEAVY_STAGE = 80
LORA_MODEL_PATHS = {"loras", "lora", "locon", "lycoris"}
PREREQ_GROUPS = {"text_encoders", "vae"}


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
    categories = _category_names(category)
    if not categories:
        return True
    for cat in categories:
        env_var = "DOWNLOAD_" + re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_").upper()
        if os.environ.get(env_var, "").strip().lower() in TRUE_VALUES:
            return True
    return False


def _category_names(category: str) -> list[str]:
    return [c.strip() for c in category.split(",") if c.strip()]


def _startup_concurrency() -> int:
    try:
        return max(1, int(os.environ.get(STARTUP_CONCURRENCY_ENV, "3")))
    except (TypeError, ValueError):
        print(f"[HF Downloader Startup] Invalid {STARTUP_CONCURRENCY_ENV}; using 3.")
        return 3


def _coerce_startup_priority(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[HF Downloader Startup] Ignoring invalid startup_priority: {value!r}")
        return None


def _assign_startup_stages(items: list[dict]) -> list[dict]:
    """Order by path group: prereqs → base models → LoRAs → everything else.
    Individual entries can override with startup_priority in the TOML."""
    ordered_items = []
    for index, item in enumerate(items):
        group = item["target_rel_path"].strip().strip("/").split("/", 1)[0].lower()
        startup_priority = _coerce_startup_priority(item.get("startup_priority"))
        if startup_priority is not None:
            stage = startup_priority
        elif group in PREREQ_GROUPS:
            stage = PREREQ_STAGE
        elif group == "diffusion_models":
            stage = BASE_MODEL_STAGE
        elif group in LORA_MODEL_PATHS:
            stage = LORA_STAGE
        else:
            stage = HEAVY_STAGE
        staged_item = dict(item)
        staged_item["_startup_stage"] = stage
        staged_item["_startup_index"] = index
        ordered_items.append(staged_item)
    return sorted(ordered_items, key=lambda item: (item["_startup_stage"], item["_startup_index"]))


_SPLIT_RE = re.compile(r"^(.+?)-(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)


def _expand_split_shards(filename: str) -> list[str]:
    """If filename matches the split pattern (e.g. model-00001-of-00003.safetensors),
    return the full list of shard filenames. Otherwise return [filename] unchanged."""
    m = _SPLIT_RE.match(filename)
    if not m:
        return [filename]
    base, _, total = m.group(1), int(m.group(2)), int(m.group(3))
    width = len(m.group(3))
    return [f"{base}-{i:0{width}d}-of-{total:0{width}d}.safetensors" for i in range(1, total + 1)]


def _task_id(item: dict) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{item['repo_id']}_{item['repo_path']}")
    return f"hf_startup_{safe}"


def _stage_label(stage: int) -> str:
    labels = {
        PREREQ_STAGE: "text encoders / VAEs",
        BASE_MODEL_STAGE: "base diffusion models",
        LORA_STAGE: "LoRAs",
        HEAVY_STAGE: "remaining models",
    }
    return labels.get(stage, f"priority {stage}")


async def _run_startup_item(
    task_id: str,
    repo_id: str,
    model_path: str,
    files: list[str],
    output_dir: str,
    output_name: str,
    name: str,
    run_download_task,
) -> None:
    await run_download_task(
        task_id=task_id,
        repo_id=repo_id,
        model_path=model_path,
        files=files,
        output_dir=output_dir,
        output_name=output_name,
        name=name,
    )


async def _run_stage(stage: int, items: list[dict], semaphore: asyncio.Semaphore, download_tasks, run_download_task) -> None:
    print(f"[HF Downloader Startup] Starting {len(items)} {_stage_label(stage)} download(s).")

    async def gated(item: dict) -> None:
        async with semaphore:
            await _run_startup_item(
                task_id=item["task_id"],
                repo_id=item["repo_id"],
                model_path=item["folder_path"],
                files=item["files"],
                output_dir=item["output_dir"],
                output_name=item["output_name"],
                name=item["name"],
                run_download_task=run_download_task,
            )

    tasks = []
    for item in items:
        task = asyncio.create_task(gated(item))
        download_tasks[item["task_id"]] = task
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(items, results):
        if isinstance(result, Exception):
            print(f"[HF Downloader Startup] Error in {item['name']}: {result}")


async def _run_startup_queue(items: list[dict], download_tasks, run_download_task) -> None:
    semaphore = asyncio.Semaphore(_startup_concurrency())
    stage_order = []
    stage_items_by_id = {}
    for item in items:
        stage = item["_startup_stage"]
        if stage not in stage_items_by_id:
            stage_order.append(stage)
            stage_items_by_id[stage] = []
        stage_items_by_id[stage].append(item)

    for stage in stage_order:
        await _run_stage(stage, stage_items_by_id[stage], semaphore, download_tasks, run_download_task)


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
            "category": category,
            "startup_priority": entry.get("startup_priority"),
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

    items = _assign_startup_stages(items)
    queued_items = []
    for item in items:
        target_path = Path(models_dir) / item["target_rel_path"]
        if target_path.exists():
            continue

        repo_path = item["repo_path"]
        if "/" in repo_path:
            folder_path, filename = repo_path.rsplit("/", 1)
        else:
            folder_path, filename = "root", repo_path

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

        download_progress[task_id] = {
            "status": "queued",
            "stage": "queued",
            "current": 0,
            "total": 0,
            "message": "Waiting to download...",
            "name": name,
            "is_startup": True,
        }

        prepared_item = dict(item)
        prepared_item.update({
            "task_id": task_id,
            "name": name,
            "folder_path": folder_path,
            "files": _expand_split_shards(filename),
            "output_dir": output_dir,
            "output_name": output_name,
        })
        queued_items.append(prepared_item)
        print(f"[HF Downloader Startup] Queued: {name}")

    if queued_items:
        future = asyncio.run_coroutine_threadsafe(
            _run_startup_queue(queued_items, download_tasks, run_download_task),
            loop,
        )
        download_tasks["hf_startup_coordinator"] = future
        print(f"[HF Downloader Startup] {len(queued_items)} model(s) queued for staged background download.")
    else:
        print("[HF Downloader Startup] All models already present, nothing to download.")
