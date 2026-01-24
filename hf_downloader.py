"""
Core HuggingFace downloader logic
Handles repo scanning, split detection, downloading, and merging
"""

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class HFDownloader:
    """Downloads and merges split safetensor files and single GGUF files from HuggingFace repos"""

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if self.hf_token else HfApi()

    def _extract_gguf_quant(self, filename: str) -> str | None:
        """
        Extract GGUF quantization string from filename if present.
        """
        if not filename.lower().endswith(".gguf"):
            return None

        stem = Path(filename).stem
        match = re.search(
            r"(?:[._-])((?:q|iq)\d[A-Za-z0-9_]*|(?:f|bf)\d[A-Za-z0-9_]*)$",
            stem,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        return None

    def _split_gguf_name(self, filename: str) -> tuple[str, str | None]:
        """
        Split a GGUF filename into base name and quant.
        """
        stem = Path(filename).stem
        quant = self._extract_gguf_quant(filename)
        if not quant:
            return stem, None

        base_name = re.sub(rf"[._-]{re.escape(quant)}$", "", stem, flags=re.IGNORECASE)
        return base_name, quant

    def _split_safetensors_name(self, filename: str) -> tuple[str, str | None]:
        """
        Split a safetensors filename into base name and variant suffix (e.g., fp16, fp8_e4m3fn).
        """
        stem = Path(filename).stem
        variant_pattern = re.compile(
            r"(?:[._-])((?:fp|bf)\d+[a-z0-9_]*|int\d+|bnb[-_]?4bit|bnb[-_]?8bit|nf4|nvfp4)$",
            re.IGNORECASE,
        )
        match = variant_pattern.search(stem)
        if not match:
            return stem, None

        raw_variant = match.group(1)
        base_name = re.sub(rf"[._-]{re.escape(raw_variant)}$", "", stem, flags=re.IGNORECASE)
        return base_name, raw_variant.lower()

    def _list_repo_files_info(self, repo_id: str) -> list:
        """
        Return file metadata with path and size across hf hub versions.
        """
        if hasattr(self.api, "list_files_info"):
            return list(self.api.list_files_info(repo_id))

        # Fallback for newer huggingface_hub versions where list_files_info was removed.
        try:
            tree = self.api.list_repo_tree(repo_id, repo_type="model", recursive=True)
        except TypeError:
            # Older versions may not accept repo_type; fall back to defaults.
            tree = self.api.list_repo_tree(repo_id, recursive=True)

        files = []
        for entry in tree:
            if getattr(entry, "type", None) == "folder":
                continue
            path = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
            if not path:
                continue
            size = getattr(entry, "size", None)
            if size is None:
                size = 0
            files.append(SimpleNamespace(rfilename=path, size=size))
        return files

    def scan_repo(self, repo_id: str) -> list[dict]:
        """
        Scan a HuggingFace repo for safetensor and GGUF files

        Returns list of model groups with metadata:
        - path: subfolder path
        - files: list of filenames with metadata (name, size, precision)
        - is_split: whether files are split into shards
        - file_count: number of files
        - total_size: total size in bytes
        - suggested_name: suggested output filename
        """
        try:
            logger.info(f"Scanning HuggingFace repo: {repo_id}")

            # Use list_files_info to get file metadata including sizes
            files_info = self._list_repo_files_info(repo_id)

            # Group safetensor files by base name and split pattern
            safetensor_groups = {}
            gguf_models = {}
            split_pattern = re.compile(r"^(.+?)-(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)

            for file_info in files_info:
                f = file_info.rfilename
                if f.lower().endswith(".safetensors"):
                    parts = f.split("/")
                    if len(parts) > 1:
                        folder = "/".join(parts[:-1])
                        filename = parts[-1]
                    else:
                        folder = "root"
                        filename = f

                    split_match = split_pattern.match(filename)
                    variant = None
                    if split_match:
                        base_name = split_match.group(1)
                        group_key = ("split", folder, base_name)
                        if group_key not in safetensor_groups:
                            safetensor_groups[group_key] = {
                                "folder": folder,
                                "base_name": base_name,
                                "files": [],
                                "is_split": True,
                                "split_info": {
                                    "total": int(split_match.group(3)),
                                    "pattern": split_pattern.pattern,
                                },
                            }
                    else:
                        base_name, variant = self._split_safetensors_name(filename)
                        group_key = ("single", folder, base_name)
                        if group_key not in safetensor_groups:
                            safetensor_groups[group_key] = {
                                "folder": folder,
                                "base_name": base_name,
                                "files": [],
                                "is_split": False,
                                "split_info": None,
                            }

                    # Extract precision from filename
                    precision = self._extract_precision(filename)

                    # Store file metadata
                    file_meta = {"name": filename, "size": file_info.size, "precision": precision}
                    if variant:
                        file_meta["variant"] = variant
                    safetensor_groups[group_key]["files"].append(file_meta)
                elif f.lower().endswith(".gguf"):
                    parts = f.split("/")
                    if len(parts) > 1:
                        folder = "/".join(parts[:-1])
                        filename = parts[-1]
                    else:
                        folder = "root"
                        filename = f

                    base_name, quant = self._split_gguf_name(filename)
                    group_key = (folder, base_name)
                    if group_key not in gguf_models:
                        gguf_models[group_key] = []

                    gguf_models[group_key].append(
                        {"name": filename, "size": file_info.size, "quant": quant}
                    )

            # Build result list for safetensors
            result = []
            for group in safetensor_groups.values():
                folder = group["folder"]
                files_list = sorted(group["files"], key=lambda x: x["name"])
                is_split = group["is_split"]
                split_info = group["split_info"]
                file_count = len(files_list)

                if is_split:
                    total_size = sum(f["size"] for f in files_list)
                else:
                    total_size = files_list[0]["size"] if files_list else 0

                precision_values = {f["precision"] for f in files_list if f.get("precision")}
                precision = None
                if len(precision_values) == 1:
                    precision = next(iter(precision_values))
                elif len(precision_values) > 1:
                    precision = "mixed"

                if is_split:
                    filenames = [f["name"] for f in files_list]
                    suggested_name = self._suggest_name(repo_id, folder, filenames, split_info)
                else:
                    if file_count == 1 and files_list:
                        suggested_name = Path(files_list[0]["name"]).stem
                    else:
                        suggested_name = group["base_name"] or (
                            Path(files_list[0]["name"]).stem if files_list else "model"
                        )

                result.append(
                    {
                        "path": folder,
                        "files": files_list,
                        "is_split": is_split,
                        "split_info": split_info,
                        "file_count": file_count,
                        "total_size": total_size,
                        "suggested_name": suggested_name,
                        "precision": precision,
                        "file_type": "safetensors",
                        "base_name": group["base_name"],
                    }
                )

            for (folder, base_name), files_list in gguf_models.items():
                sorted_files = sorted(files_list, key=lambda x: x["name"])
                display_size = sorted_files[0]["size"] if sorted_files else 0
                quant_options = sorted(
                    {f["quant"] for f in files_list if f.get("quant") is not None}
                )
                result.append(
                    {
                        "path": folder,
                        "files": sorted_files,
                        "is_split": False,
                        "split_info": None,
                        "file_count": len(files_list),
                        "total_size": display_size,
                        "suggested_name": base_name or self._suggest_name(repo_id, folder, [], None),
                        "precision": None,
                        "file_type": "gguf",
                        "quant_options": quant_options,
                        "base_name": base_name,
                    }
                )

            logger.info(f"Found {len(result)} model group(s) in repo")
            return result

        except Exception as e:
            logger.error(f"Error scanning repo {repo_id}: {e}")
            raise

    def _extract_precision(self, filename: str) -> str | None:
        """
        Extract precision type from filename (fp16, fp32, bf16, etc.)
        Returns precision string if found, None otherwise
        """
        # Common precision patterns
        precision_pattern = re.compile(
            r"[-_]((?:fp|bf)\d+[a-z0-9_]*|int\d+|bnb[-_]?4bit|bnb[-_]?8bit|nf4|nvfp4)",
            re.IGNORECASE,
        )
        match = precision_pattern.search(filename)
        if match:
            return match.group(1).lower().replace("_", "-")
        return None

    def _detect_splits(self, files: list[str]) -> dict | None:
        """
        Detect if files follow split pattern like model-00001-of-00003.safetensors
        Returns dict with 'total' count if split pattern detected, None otherwise
        """
        pattern = re.compile(r"-(\d+)-of-(\d+)\.safetensors$")

        for f in files:
            match = pattern.search(f)
            if match:
                total = int(match.group(2))
                return {"total": total, "pattern": pattern.pattern}

        return None

    def _suggest_name(
        self, repo_id: str, folder: str, files: list[str], split_info: dict | None
    ) -> str:
        """
        Suggest an output filename based on repo structure

        Priority:
        1. For root files: use actual filename (without .safetensors, preserving precision)
        2. For folders: check config.json for _name_or_path, else use folder name
        3. For split files: extract base name from pattern
        4. Fallback: repo name
        """
        # Root folder with single file - use the actual filename
        if folder == "root" and len(files) == 1:
            # Strip extension but keep precision suffix
            return Path(files[0]).stem

        # If split files, try to extract base name
        if split_info and files:
            # Remove the split suffix to get base name
            match = re.search(r"^(.+?)-\d+-of-\d+\.safetensors$", files[0])
            if match:
                base_name = match.group(1)
                # For folders, also check config.json for better naming
                if folder != "root":
                    config_name = self._get_name_from_config(repo_id, folder)
                    if config_name:
                        return config_name
                return base_name

        # If folder has a meaningful name, try config.json first
        if folder != "root" and folder:
            # Try to get name from config.json
            config_name = self._get_name_from_config(repo_id, folder)
            if config_name:
                return config_name

            # Fall back to folder name (preserve precision suffix)
            name = folder.split("/")[-1]
            return name

        # Fall back to repo name
        repo_name = repo_id.split("/")[-1]
        return repo_name

    def _get_name_from_config(self, repo_id: str, folder: str) -> str | None:
        """
        Try to extract model name from config.json in the folder
        Returns the _name_or_path value or model_type if found
        """
        try:
            config_path = f"{folder}/config.json"

            # Download and parse config.json
            local_config = hf_hub_download(repo_id=repo_id, filename=config_path)

            with Path(local_config).open() as f:
                config = json.load(f)

            # Try _name_or_path first (e.g., "google/t5-v1_1-xxl")
            if "_name_or_path" in config:
                name = config["_name_or_path"]
                # Extract just the model name (e.g., "t5-v1_1-xxl" from "google/t5-v1_1-xxl")
                if "/" in name:
                    return name.split("/")[-1]
                return name

            # Fallback to model_type if available
            if "model_type" in config:
                return config["model_type"]

        except Exception as e:
            logger.debug(f"Could not read config.json from {folder}: {e}")

        return None

    def download_and_merge(
        self,
        repo_id: str,
        folder_path: str,
        files: list[str],
        output_dir: str,
        output_name: str,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> str:
        """
        Download files from HuggingFace and merge if split

        Args:
            repo_id: HuggingFace repo identifier (e.g., 'username/model')
            folder_path: Subfolder path within repo
            files: List of files to download
            output_dir: Destination directory
            output_name: Output filename (without extension)
            progress_callback: Optional callback(stage, current, total, message)

        Returns:
            Path to the merged/downloaded file
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Check if HF CLI is available
            if not self._check_hf_cli():
                raise RuntimeError(
                    "HF CLI not found. Please install with: curl -LsSf https://hf.co/cli/install.sh | bash"
                )

            # Login to HF if token available
            if self.hf_token:
                if progress_callback:
                    progress_callback("auth", 0, 1, "Authenticating with HuggingFace...")
                self._hf_login()

            # Download files
            if progress_callback:
                progress_callback("download", 0, len(files), f"Downloading {len(files)} file(s)...")

            downloaded_paths = []
            for idx, filename in enumerate(files):
                file_path = f"{folder_path}/{filename}" if folder_path != "root" else filename

                if progress_callback:
                    progress_callback("download", idx, len(files), f"Downloading {filename}...")

                cached_path = self._download_file(repo_id, file_path)
                downloaded_paths.append(cached_path)

            if progress_callback:
                progress_callback("download", len(files), len(files), "Download complete")

            # Merge if multiple files, otherwise just copy
            output_ext = Path(files[0]).suffix if files else ".safetensors"
            if not output_ext:
                output_ext = ".safetensors"
            output_ext = output_ext.lower()
            output_path = str(Path(output_dir) / f"{output_name}{output_ext}")

            if len(files) > 1:
                if output_ext != ".safetensors":
                    raise RuntimeError("Only safetensors files can be merged.")
                if progress_callback:
                    progress_callback("merge", 0, len(files), "Starting merge...")

                self._merge_files(downloaded_paths, output_path, progress_callback)

                if progress_callback:
                    progress_callback("merge", len(files), len(files), "Merge complete")
            else:
                # Single file - prefer symlink from cache for faster loads
                if progress_callback:
                    progress_callback("copy", 0, 1, "Linking file from cache...")

                output_path_obj = Path(output_path)
                cached_path = Path(downloaded_paths[0])
                try:
                    if output_path_obj.exists() or output_path_obj.is_symlink():
                        if output_path_obj.is_dir():
                            raise RuntimeError(f"Output path is a directory: {output_path}")
                        output_path_obj.unlink()
                    output_path_obj.symlink_to(cached_path)
                    action = "Linked"
                except OSError as e:
                    logger.warning(f"Symlink failed, falling back to copy: {e}")
                    shutil.copy2(cached_path, output_path_obj)
                    action = "Copied"

                if progress_callback:
                    progress_callback("copy", 1, 1, f"{action} file")

            logger.info(f"Successfully saved model to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error downloading/merging model: {e}")
            raise

    def _check_hf_cli(self) -> bool:
        """Check if HF CLI is installed"""
        try:
            subprocess.run(["hf", "version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _hf_login(self):
        """Login to HuggingFace CLI with token"""
        try:
            subprocess.run(
                ["hf", "auth", "login", "--token", self.hf_token],
                capture_output=True,
                check=True,
                text=True,
            )
            logger.info("Successfully authenticated with HuggingFace")
        except subprocess.CalledProcessError as e:
            logger.warning(f"HF auth warning: {e.stderr}")
            # Don't fail - token might already be cached

    def _download_file(self, repo_id: str, file_path: str) -> str:
        """
        Download a single file using HF CLI
        Returns path to cached file
        """
        try:
            result = subprocess.run(
                ["hf", "download", repo_id, file_path], capture_output=True, check=True, text=True
            )
            cached_path = result.stdout.strip()
            logger.info(f"Downloaded {file_path} to {cached_path}")
            return cached_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading {file_path}: {e.stderr}")
            raise

    def _merge_files(
        self,
        file_paths: list[str],
        output_path: str,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> None:
        """
        Merge multiple safetensor files into one
        Uses the proven 3-line merge approach
        """
        try:
            logger.info(f"Merging {len(file_paths)} safetensor files...")

            # Load all shards
            tensors = {}
            for idx, shard_path in enumerate(sorted(file_paths)):
                logger.info(f"Loading shard {idx + 1}/{len(file_paths)}: {Path(shard_path).name}")
                if progress_callback:
                    progress_callback(
                        "merge",
                        idx,
                        len(file_paths),
                        f"Loading shard {idx + 1}/{len(file_paths)}...",
                    )
                tensors.update(load_file(shard_path))

            logger.info(f"Loaded {len(tensors)} tensors total, saving merged file...")
            if progress_callback:
                progress_callback(
                    "merge", len(file_paths), len(file_paths) + 1, "Saving merged file..."
                )

            # Save as single file
            save_file(tensors, output_path)

            # Get file size for logging
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"Saved merged file: {output_path} ({size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"Error merging files: {e}")
            raise
