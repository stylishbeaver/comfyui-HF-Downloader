"""
API routes for HuggingFace Downloader extension
Integrates with ComfyUI's server
"""

import asyncio
import logging
import traceback
from pathlib import Path

import folder_paths
import server
from aiohttp import web

# Handle both package and standalone imports
try:
    from .hf_downloader import HFDownloader
except ImportError:
    from hf_downloader import HFDownloader

logger = logging.getLogger(__name__)

# Get ComfyUI server instance
prompt_server = server.PromptServer.instance

# Global state for download tasks
download_tasks = {}
download_progress = {}


def get_model_dir(model_type: str) -> str:
    """
    Get the appropriate model directory based on type
    Uses ComfyUI's folder_paths to respect user configuration
    """
    type_mapping = {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "vae": "vae",
        "upscale_model": "upscale_models",
        "embedding": "embeddings",
        "clip": "clip",
        "controlnet": "controlnet",
        "diffusion_model": "diffusion_models",
        "text_encoder": "text_encoders",
    }

    folder_name = type_mapping.get(model_type, "checkpoints")

    # OVERRIDE: ComfyUI's folder_paths incorrectly maps diffusion_models to unet/
    # But nodes actually look in diffusion_models/, so use direct path for this
    if folder_name == "diffusion_models":
        return str(Path(folder_paths.models_dir) / "diffusion_models")

    # Get directory from folder_paths
    if folder_name in folder_paths.folder_names_and_paths:
        return folder_paths.folder_names_and_paths[folder_name][0][0]
    else:
        # Fallback to models dir
        return str(Path(folder_paths.models_dir) / folder_name)


@prompt_server.routes.post("/hf_downloader/scan")
async def scan_repo_handler(request):
    """
    Scan a HuggingFace repo for safetensor and GGUF files

    POST /hf_downloader/scan
    Body: {"repo_id": "username/model"}
    """
    try:
        data = await request.json()
        repo_id = data.get("repo_id", "").strip()

        if not repo_id:
            return web.json_response({"error": "repo_id is required"}, status=400)

        # Validate repo_id format (should be username/model or org/model)
        if "/" not in repo_id:
            return web.json_response(
                {"error": "Invalid repo_id format. Expected: username/model"}, status=400
            )

        downloader = HFDownloader()
        models = await asyncio.to_thread(downloader.scan_repo, repo_id)

        return web.json_response({"success": True, "repo_id": repo_id, "models": models})

    except Exception as e:
        logger.error(f"Error scanning repo: {e}")
        logger.error(traceback.format_exc())
        return web.json_response({"error": str(e)}, status=500)


@prompt_server.routes.post("/hf_downloader/download")
async def download_model_handler(request):
    """
    Start downloading and merging a model

    POST /hf_downloader/download
    Body: {
        "repo_id": "username/model",
        "model_path": "subfolder",
        "files": ["file1.safetensors", "file2.gguf", ...],
        "output_name": "model_name",
        "model_type": "checkpoint"
    }
    """
    try:
        data = await request.json()

        repo_id = data.get("repo_id", "").strip()
        model_path = data.get("model_path", "root")
        files = data.get("files", [])
        output_name = data.get("output_name", "model")
        model_type = data.get("model_type", "checkpoint")

        # Validation
        if not repo_id or not files or not output_name:
            return web.json_response(
                {"error": "repo_id, files, and output_name are required"}, status=400
            )

        # Generate unique task ID
        task_id = f"{repo_id.replace('/', '_')}_{output_name}"

        # Check if task already running
        if task_id in download_tasks and not download_tasks[task_id].done():
            return web.json_response(
                {"error": "Download already in progress for this model"}, status=409
            )

        # Get output directory
        output_dir = get_model_dir(model_type)

        # Start download task in background
        task = asyncio.create_task(
            run_download_task(
                task_id=task_id,
                repo_id=repo_id,
                model_path=model_path,
                files=files,
                output_dir=output_dir,
                output_name=output_name,
            )
        )

        download_tasks[task_id] = task
        download_progress[task_id] = {
            "status": "starting",
            "stage": "init",
            "current": 0,
            "total": 0,
            "message": "Initializing download...",
        }

        return web.json_response(
            {"success": True, "task_id": task_id, "message": "Download started"}
        )

    except Exception as e:
        logger.error(f"Error starting download: {e}")
        logger.error(traceback.format_exc())
        return web.json_response({"error": str(e)}, status=500)


@prompt_server.routes.get("/hf_downloader/progress/{task_id}")
async def get_progress_handler(request):
    """
    Get progress of a download task

    GET /hf_downloader/progress/<task_id>
    """
    try:
        task_id = request.match_info.get("task_id")

        if task_id not in download_progress:
            return web.json_response({"error": "Task not found"}, status=404)

        progress = download_progress[task_id]

        # Check if task is done
        if task_id in download_tasks:
            task = download_tasks[task_id]
            if task.done():
                if task.exception():
                    progress["status"] = "error"
                    progress["message"] = str(task.exception())
                elif progress["status"] != "completed":
                    progress["status"] = "completed"

        return web.json_response(progress)

    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def run_download_task(
    task_id: str,
    repo_id: str,
    model_path: str,
    files: list[str],
    output_dir: str,
    output_name: str,
) -> None:
    """
    Background task to run the download and merge operation
    """

    def progress_callback(stage, current, total, message):
        """Update progress state"""
        download_progress[task_id] = {
            "status": "running",
            "stage": stage,
            "current": current,
            "total": total,
            "message": message,
        }

    try:
        download_progress[task_id]["status"] = "running"

        downloader = HFDownloader()

        # Run download in thread pool to avoid blocking
        output_path = await asyncio.to_thread(
            downloader.download_and_merge,
            repo_id=repo_id,
            folder_path=model_path,
            files=files,
            output_dir=output_dir,
            output_name=output_name,
            progress_callback=progress_callback,
        )

        download_progress[task_id] = {
            "status": "completed",
            "stage": "done",
            "current": 1,
            "total": 1,
            "message": f"Successfully saved to {output_path}",
            "output_path": output_path,
        }

        logger.info(f"Download task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Download task {task_id} failed: {e}")
        logger.error(traceback.format_exc())

        download_progress[task_id] = {
            "status": "error",
            "stage": "error",
            "current": 0,
            "total": 0,
            "message": str(e),
        }


@prompt_server.routes.post("/hf_downloader/abort/{task_id}")
async def abort_download_handler(request):
    """
    Abort a running download task

    POST /hf_downloader/abort/<task_id>
    """
    try:
        task_id = request.match_info["task_id"]

        if task_id not in download_tasks:
            return web.json_response({"error": "Task not found"}, status=404)

        task = download_tasks[task_id]

        if not task.done():
            task.cancel()
            download_progress[task_id] = {
                "status": "cancelled",
                "stage": "cancelled",
                "current": 0,
                "total": 0,
                "message": "Download cancelled by user",
            }
            logger.info(f"Cancelled download task: {task_id}")

        return web.json_response({"success": True, "message": "Download cancelled"})

    except Exception as e:
        logger.error(f"Error aborting download: {e}")
        return web.json_response({"error": str(e)}, status=500)


@prompt_server.routes.get("/hf_downloader/files/{model_type}")
async def list_files_handler(request):
    """
    List downloaded model files by type

    GET /hf_downloader/files/<model_type>
    """
    try:
        model_type = request.match_info["model_type"]
        model_dir = Path(get_model_dir(model_type))

        if not model_dir.exists():
            return web.json_response({"files": []})

        files = []
        for filepath in model_dir.iterdir():
            if filepath.suffix in {".safetensors", ".gguf"}:
                stat = filepath.stat()
                files.append(
                    {
                        "name": filepath.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "path": str(filepath),
                    }
                )

        # Sort by modified time, newest first
        files.sort(key=lambda x: x["modified"], reverse=True)

        return web.json_response({"files": files})

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return web.json_response({"error": str(e)}, status=500)


@prompt_server.routes.delete("/hf_downloader/files")
async def delete_file_handler(request):
    """
    Delete a model file

    DELETE /hf_downloader/files
    Body: { "filepath": "/path/to/file.safetensors" }
    """
    try:
        data = await request.json()
        filepath = data.get("filepath")

        if not filepath:
            return web.json_response({"error": "filepath is required"}, status=400)

        # Security: ensure file is in models directory
        models_dir = Path(folder_paths.models_dir).resolve()
        file_path = Path(filepath).resolve()

        if not str(file_path).startswith(str(models_dir)):
            return web.json_response({"error": "Invalid file path"}, status=403)

        if not file_path.exists():
            return web.json_response({"error": "File not found"}, status=404)

        file_path.unlink()
        logger.info(f"Deleted file: {filepath}")

        return web.json_response({"success": True, "message": "File deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return web.json_response({"error": str(e)}, status=500)


# Routes are registered via decorators when this module is imported
logger.info("HF Downloader routes registered successfully")
