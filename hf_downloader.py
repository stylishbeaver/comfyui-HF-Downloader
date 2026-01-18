"""
Core HuggingFace downloader logic
Handles repo scanning, split detection, downloading, and merging
"""

import os
import re
import glob
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class HFDownloader:
    """Downloads and merges split safetensor files from HuggingFace repos"""

    def __init__(self):
        self.hf_token = os.getenv('HF_TOKEN')
        self.api = HfApi(token=self.hf_token) if self.hf_token else HfApi()

    def scan_repo(self, repo_id: str) -> List[Dict]:
        """
        Scan a HuggingFace repo for safetensor files

        Returns list of model groups with metadata:
        - path: subfolder path
        - files: list of filenames
        - is_split: whether files are split into shards
        - file_count: number of files
        - total_size: estimated total size in bytes
        - suggested_name: suggested output filename
        """
        try:
            logger.info(f"Scanning HuggingFace repo: {repo_id}")
            files = self.api.list_repo_files(repo_id)

            # Group safetensor files by subfolder
            models = {}
            file_info = {}

            for f in files:
                if f.endswith('.safetensors'):
                    parts = f.split('/')
                    if len(parts) > 1:
                        folder = '/'.join(parts[:-1])
                        filename = parts[-1]
                    else:
                        folder = 'root'
                        filename = f

                    if folder not in models:
                        models[folder] = []
                    models[folder].append(filename)

            # Analyze each group and detect splits
            result = []
            for folder, files_list in models.items():
                split_info = self._detect_splits(files_list)
                suggested_name = self._suggest_name(repo_id, folder, files_list, split_info)

                result.append({
                    'path': folder,
                    'files': sorted(files_list),
                    'is_split': split_info is not None,
                    'split_info': split_info,
                    'file_count': len(files_list),
                    'suggested_name': suggested_name
                })

            logger.info(f"Found {len(result)} model group(s) in repo")
            return result

        except Exception as e:
            logger.error(f"Error scanning repo {repo_id}: {e}")
            raise

    def _detect_splits(self, files: List[str]) -> Optional[Dict]:
        """
        Detect if files follow split pattern like model-00001-of-00003.safetensors
        Returns dict with 'total' count if split pattern detected, None otherwise
        """
        pattern = re.compile(r'-(\d+)-of-(\d+)\.safetensors$')

        for f in files:
            match = pattern.search(f)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                return {
                    'total': total,
                    'pattern': pattern.pattern
                }

        return None

    def _suggest_name(self, repo_id: str, folder: str, files: List[str], split_info: Optional[Dict]) -> str:
        """
        Suggest an output filename based on repo structure

        Priority:
        1. Use folder name if not 'root'
        2. Use repo name
        3. Extract base name from split files
        """
        # If folder has a meaningful name, use it
        if folder != 'root' and folder:
            # Use last part of folder path
            name = folder.split('/')[-1]
            # Clean up common suffixes
            name = re.sub(r'[-_](fp16|fp32|bf16|bnb|4bit|8bit)$', '', name, flags=re.IGNORECASE)
            return name

        # If split files, try to extract base name
        if split_info and files:
            # Remove the split suffix to get base name
            match = re.search(r'^(.+?)-\d+-of-\d+\.safetensors$', files[0])
            if match:
                return match.group(1)

        # Fall back to repo name
        repo_name = repo_id.split('/')[-1]
        return repo_name

    def download_and_merge(
        self,
        repo_id: str,
        folder_path: str,
        files: List[str],
        output_dir: str,
        output_name: str,
        progress_callback: Optional[callable] = None
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
            os.makedirs(output_dir, exist_ok=True)

            # Check if HF CLI is available
            if not self._check_hf_cli():
                raise RuntimeError("HF CLI not found. Please install with: curl -LsSf https://hf.co/cli/install.sh | bash")

            # Login to HF if token available
            if self.hf_token:
                if progress_callback:
                    progress_callback('auth', 0, 1, 'Authenticating with HuggingFace...')
                self._hf_login()

            # Download files
            if progress_callback:
                progress_callback('download', 0, len(files), f'Downloading {len(files)} file(s)...')

            downloaded_paths = []
            for idx, filename in enumerate(files):
                file_path = f"{folder_path}/{filename}" if folder_path != 'root' else filename

                if progress_callback:
                    progress_callback('download', idx, len(files), f'Downloading {filename}...')

                cached_path = self._download_file(repo_id, file_path)
                downloaded_paths.append(cached_path)

            if progress_callback:
                progress_callback('download', len(files), len(files), 'Download complete')

            # Merge if multiple files, otherwise just copy
            output_path = os.path.join(output_dir, f"{output_name}.safetensors")

            if len(files) > 1:
                if progress_callback:
                    progress_callback('merge', 0, 1, 'Merging safetensor files...')

                self._merge_files(downloaded_paths, output_path)

                if progress_callback:
                    progress_callback('merge', 1, 1, 'Merge complete')
            else:
                # Single file - just copy from cache
                if progress_callback:
                    progress_callback('copy', 0, 1, 'Copying file...')

                import shutil
                shutil.copy2(downloaded_paths[0], output_path)

                if progress_callback:
                    progress_callback('copy', 1, 1, 'Copy complete')

            logger.info(f"Successfully saved model to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error downloading/merging model: {e}")
            raise

    def _check_hf_cli(self) -> bool:
        """Check if HF CLI is installed"""
        try:
            subprocess.run(['hf', 'version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _hf_login(self):
        """Login to HuggingFace CLI with token"""
        try:
            subprocess.run(
                ['hf', 'auth', 'login', '--token', self.hf_token],
                capture_output=True,
                check=True,
                text=True
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
                ['hf', 'download', repo_id, file_path],
                capture_output=True,
                check=True,
                text=True
            )
            cached_path = result.stdout.strip()
            logger.info(f"Downloaded {file_path} to {cached_path}")
            return cached_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading {file_path}: {e.stderr}")
            raise

    def _merge_files(self, file_paths: List[str], output_path: str):
        """
        Merge multiple safetensor files into one
        Uses the proven 3-line merge approach
        """
        try:
            logger.info(f"Merging {len(file_paths)} safetensor files...")

            # Load all shards
            tensors = {}
            for shard_path in sorted(file_paths):
                logger.debug(f"Loading {shard_path}")
                tensors.update(load_file(shard_path))

            logger.info(f"Loaded {len(tensors)} tensors total")

            # Save as single file
            save_file(tensors, output_path)

            # Get file size for logging
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Saved merged file: {output_path} ({size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"Error merging files: {e}")
            raise
