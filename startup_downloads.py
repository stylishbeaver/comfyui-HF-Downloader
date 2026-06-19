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
import time
from pathlib import Path
from typing import Optional

TOML_PATH = "/workspace/hf_models.toml"
SKIP_ENV = "SKIP_MODEL_DOWNLOADS"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
STARTUP_CONCURRENCY_ENV = "HF_STARTUP_MAX_CONCURRENT"
STAGE_DIR_ENV = "MODEL_DOWNLOAD_STAGE_DIR"
DEFAULT_STAGE_DIR = "/tmp/comfyui-model-download-stages/default"
HF_PREREQS_DONE_MARKER = "hf-prereqs.done"
HF_LORAS_DONE_MARKER = "hf-loras.done"
CIVITAI_LORAS_DONE_MARKER = "civitai-loras.done"
VAE_STAGE = 0
Z_IMAGE_TURBO_STAGE = 10
FLUX_KLEIN_9B_STAGE = 20
HERETIC_TEXT_ENCODER_STAGE = 30
LORA_STAGE = 40
HEAVY_STAGE = 80
LORA_MODEL_PATHS = {"loras", "lora", "locon", "lycoris"}


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


def _target_group(target_rel_path: str) -> str:
    return target_rel_path.strip().strip("/").split("/", 1)[0].lower()


def _has_category(item: dict, category: str) -> bool:
    return category in _category_names(item.get("category", ""))


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


def _stage_dir() -> Path:
    path = Path(os.environ.get(STAGE_DIR_ENV, DEFAULT_STAGE_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_marker(name: str) -> None:
    try:
        marker = _stage_dir() / name
        marker.write_text(str(time.time()), encoding="utf-8")
        print(f"[HF Downloader Startup] Stage marker ready: {marker}")
    except Exception as e:
        print(f"[HF Downloader Startup] Could not write stage marker {name}: {e}")


async def _wait_for_markers(names: tuple[str, ...], label: str) -> None:
    missing_logged = ""
    while True:
        missing = [name for name in names if not (_stage_dir() / name).exists()]
        if not missing:
            return
        missing_text = ", ".join(missing)
        if missing_text != missing_logged:
            print(f"[HF Downloader Startup] Waiting for {label}: {missing_text}")
            missing_logged = missing_text
        await asyncio.sleep(5)


def _assign_startup_stages(items: list[dict]) -> list[dict]:
    """Apply the first-use order: VAE, core bases, heretic text encoders, LoRAs, then the backlog."""
    ordered_items = []

    for index, item in enumerate(items):
        rel_path = item["target_rel_path"].strip().strip("/").lower()
        repo_path = item["repo_path"].strip().strip("/").lower()
        path_text = f"{rel_path} {repo_path}"
        group = _target_group(item["target_rel_path"])

        startup_priority = _coerce_startup_priority(item.get("startup_priority"))
        if startup_priority is not None:
            stage = startup_priority
        elif group == "vae":
            stage = VAE_STAGE
        elif group == "diffusion_models" and _has_category(item, "Z_IMAGE_TURBO"):
            stage = Z_IMAGE_TURBO_STAGE
        elif group == "diffusion_models" and _has_category(item, "FLUX_KLEIN") and "9b" in path_text:
            stage = FLUX_KLEIN_9B_STAGE
        elif group == "text_encoders" and ("heretic" in path_text or "abliterated" in path_text):
            stage = HERETIC_TEXT_ENCODER_STAGE
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
    """If *filename* matches the split pattern (e.g. model-00001-of-00003.safetensors),
    return the full list of shard filenames.  Otherwise return [filename] unchanged."""
    m = _SPLIT_RE.match(filename)
    if not m:
        return [filename]
    base, _, total = m.group(1), int(m.group(2)), int(m.group(3))
    width = len(m.group(3))  # preserve zero-padding width
    return [f"{base}-{i:0{width}d}-of-{total:0{width}d}.safetensors" for i in range(1, total + 1)]


def _task_id(item: dict) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{item['repo_id']}_{item['repo_path']}")
    return f"hf_startup_{safe}"


def _stage_label(stage: int) -> str:
    labels = {
        VAE_STAGE: "VAEs",
        Z_IMAGE_TURBO_STAGE: "Z-Image Turbo core",
        FLUX_KLEIN_9B_STAGE: "Flux/Klein 9B core",
        HERETIC_TEXT_ENCODER_STAGE: "heretic text encoders",
        LORA_STAGE: "LoRAs",
        HEAVY_STAGE: "remaining heavy models",
    }
    return labels.get(stage, f"custom priority {stage}")


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

    prereqs_marked = False
    loras_marked = False

    try:
        for stage in stage_order:
            if stage >= LORA_STAGE and not prereqs_marked:
                _write_marker(HF_PREREQS_DONE_MARKER)
                prereqs_marked = True

            if stage > LORA_STAGE:
                if not loras_marked:
                    _write_marker(HF_LORAS_DONE_MARKER)
                    loras_marked = True

            await _run_stage(stage, stage_items_by_id[stage], semaphore, download_tasks, run_download_task)

            if stage == LORA_STAGE and not loras_marked:
                _write_marker(HF_LORAS_DONE_MARKER)
                loras_marked = True

        if not prereqs_marked:
            _write_marker(HF_PREREQS_DONE_MARKER)
        if not loras_marked:
            _write_marker(HF_LORAS_DONE_MARKER)
    except Exception as e:
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        print(f"[HF Downloader Startup] Startup queue coordinator failed: {e}")
        raise


def start_background_downloads() -> None:
    if _should_skip():
        print("[HF Downloader Startup] SKIP_MODEL_DOWNLOADS set, skipping.")
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        return

    if not os.path.exists(TOML_PATH):
        print(f"[HF Downloader Startup] No config at {TOML_PATH}, skipping.")
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        return

    data = _parse_toml(TOML_PATH)
    if not data:
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        return

    all_items = data.get("model", [])
    if not isinstance(all_items, list):
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
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
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        return

    try:
        import folder_paths
        models_dir = folder_paths.models_dir
    except Exception:
        print("[HF Downloader Startup] Could not get models_dir from folder_paths.")
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        return

    try:
        from .server_routes import download_tasks, download_progress, run_download_task
        import server
        loop = server.PromptServer.instance.loop
    except Exception as e:
        print(f"[HF Downloader Startup] Could not access download infrastructure: {e}")
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        return

    items = _assign_startup_stages(items)
    queued_items = []
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
        _write_marker(HF_PREREQS_DONE_MARKER)
        _write_marker(HF_LORAS_DONE_MARKER)
        print("[HF Downloader Startup] All models already present, nothing to download.")
