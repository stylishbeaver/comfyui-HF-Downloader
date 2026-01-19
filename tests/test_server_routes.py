"""
Comprehensive tests for server routes
Tests API endpoints, request validation, and async operations
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from server_routes import (
    abort_download_handler,
    delete_file_handler,
    download_model_handler,
    get_model_dir,
    get_progress_handler,
    list_files_handler,
    scan_repo_handler,
)


class TestGetModelDir:
    """Test model directory resolution"""

    def test_get_model_dir_checkpoint(self):
        """Test checkpoint directory resolution"""
        result = get_model_dir("checkpoint")
        assert "checkpoints" in result

    def test_get_model_dir_lora(self):
        """Test LoRA directory resolution"""
        result = get_model_dir("lora")
        assert "loras" in result

    def test_get_model_dir_diffusion_override(self):
        """Test diffusion_models directory override"""
        result = get_model_dir("diffusion_model")
        assert "diffusion_models" in result

    def test_get_model_dir_unknown_type(self):
        """Test fallback for unknown model type"""
        result = get_model_dir("unknown_type")
        assert "checkpoints" in result  # Should fallback to checkpoints


class TestScanRepoHandler:
    """Test /hf_downloader/scan endpoint"""

    @pytest.mark.asyncio
    async def test_scan_repo_success(self):
        """Test successful repo scan"""
        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"repo_id": "user/test-repo"})

        mock_models = [
            {
                "path": "root",
                "files": [{"name": "model.safetensors", "size": 1000000}],
                "is_split": False,
                "file_count": 1,
                "total_size": 1000000,
                "suggested_name": "model",
            }
        ]

        with patch("server_routes.HFDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.scan_repo.return_value = mock_models
            mock_downloader_class.return_value = mock_downloader

            response = await scan_repo_handler(mock_request)

            assert response.status == 200
            body = response.body
            assert b"success" in body
            assert b"user/test-repo" in body

    @pytest.mark.asyncio
    async def test_scan_repo_missing_repo_id(self):
        """Test scan with missing repo_id"""
        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={})

        response = await scan_repo_handler(mock_request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_scan_repo_invalid_format(self):
        """Test scan with invalid repo_id format"""
        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"repo_id": "invalid-format"})

        response = await scan_repo_handler(mock_request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_scan_repo_api_error(self):
        """Test scan with API error"""
        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"repo_id": "user/test-repo"})

        with patch("server_routes.HFDownloader") as mock_downloader_class:
            mock_downloader = Mock()
            mock_downloader.scan_repo.side_effect = Exception("API Error")
            mock_downloader_class.return_value = mock_downloader

            response = await scan_repo_handler(mock_request)
            assert response.status == 500


class TestDownloadModelHandler:
    """Test /hf_downloader/download endpoint"""

    @pytest.mark.asyncio
    async def test_download_start_success(self):
        """Test successful download start"""
        mock_request = Mock()
        mock_request.json = AsyncMock(
            return_value={
                "repo_id": "user/test-repo",
                "model_path": "root",
                "files": ["model.safetensors"],
                "output_name": "test_model",
                "model_type": "checkpoint",
            }
        )

        response = await download_model_handler(mock_request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_download_missing_required_fields(self):
        """Test download with missing required fields"""
        mock_request = Mock()
        mock_request.json = AsyncMock(
            return_value={
                "repo_id": "user/test-repo",
                # Missing files and output_name
            }
        )

        response = await download_model_handler(mock_request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_download_already_in_progress(self):
        """Test starting download when one is already in progress"""
        from server_routes import download_tasks

        mock_request = Mock()
        mock_request.json = AsyncMock(
            return_value={
                "repo_id": "user/test-repo",
                "model_path": "root",
                "files": ["model.safetensors"],
                "output_name": "test_model",
                "model_type": "checkpoint",
            }
        )

        # Create a mock ongoing task
        task_id = "user_test-repo_test_model"
        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        download_tasks[task_id] = mock_task

        try:
            response = await download_model_handler(mock_request)
            assert response.status == 409  # Conflict
        finally:
            # Clean up
            download_tasks.pop(task_id, None)


class TestProgressHandler:
    """Test /hf_downloader/progress/<task_id> endpoint"""

    @pytest.mark.asyncio
    async def test_get_progress_running(self):
        """Test getting progress for running task"""
        from server_routes import download_progress

        mock_request = Mock()
        mock_request.match_info = {"task_id": "test_task"}

        download_progress["test_task"] = {
            "status": "running",
            "stage": "download",
            "current": 1,
            "total": 3,
            "message": "Downloading file 1/3",
        }

        try:
            response = await get_progress_handler(mock_request)
            assert response.status == 200
        finally:
            download_progress.pop("test_task", None)

    @pytest.mark.asyncio
    async def test_get_progress_not_found(self):
        """Test getting progress for non-existent task"""
        mock_request = Mock()
        mock_request.match_info = {"task_id": "nonexistent"}

        response = await get_progress_handler(mock_request)
        assert response.status == 404

    @pytest.mark.asyncio
    async def test_get_progress_completed(self):
        """Test getting progress for completed task"""
        from server_routes import download_progress, download_tasks

        mock_request = Mock()
        mock_request.match_info = {"task_id": "completed_task"}

        # Set up completed task
        mock_task = Mock()
        mock_task.done = Mock(return_value=True)
        mock_task.exception = Mock(return_value=None)
        download_tasks["completed_task"] = mock_task

        download_progress["completed_task"] = {
            "status": "running",  # Will be updated to completed
            "stage": "merge",
            "current": 3,
            "total": 3,
            "message": "Merging files",
        }

        try:
            response = await get_progress_handler(mock_request)
            assert response.status == 200
            # Status should be updated to completed
            assert download_progress["completed_task"]["status"] == "completed"
        finally:
            download_tasks.pop("completed_task", None)
            download_progress.pop("completed_task", None)

    @pytest.mark.asyncio
    async def test_get_progress_error(self):
        """Test getting progress for task with error"""
        from server_routes import download_progress, download_tasks

        mock_request = Mock()
        mock_request.match_info = {"task_id": "error_task"}

        mock_task = Mock()
        mock_task.done = Mock(return_value=True)
        mock_task.exception = Mock(return_value=Exception("Download failed"))
        download_tasks["error_task"] = mock_task

        download_progress["error_task"] = {
            "status": "running",
            "stage": "download",
            "current": 0,
            "total": 1,
            "message": "Starting download",
        }

        try:
            response = await get_progress_handler(mock_request)
            assert response.status == 200
            assert download_progress["error_task"]["status"] == "error"
        finally:
            download_tasks.pop("error_task", None)
            download_progress.pop("error_task", None)


class TestAbortHandler:
    """Test /hf_downloader/abort/<task_id> endpoint"""

    @pytest.mark.asyncio
    async def test_abort_running_task(self):
        """Test aborting a running download"""
        from server_routes import download_progress, download_tasks

        mock_request = Mock()
        mock_request.match_info = {"task_id": "running_task"}

        mock_task = Mock()
        mock_task.done = Mock(return_value=False)
        mock_task.cancel = Mock()
        download_tasks["running_task"] = mock_task

        download_progress["running_task"] = {
            "status": "running",
            "stage": "download",
            "current": 1,
            "total": 3,
        }

        try:
            response = await abort_download_handler(mock_request)
            assert response.status == 200
            mock_task.cancel.assert_called_once()
            assert download_progress["running_task"]["status"] == "cancelled"
        finally:
            download_tasks.pop("running_task", None)
            download_progress.pop("running_task", None)

    @pytest.mark.asyncio
    async def test_abort_nonexistent_task(self):
        """Test aborting non-existent task"""
        mock_request = Mock()
        mock_request.match_info = {"task_id": "nonexistent"}

        response = await abort_download_handler(mock_request)
        assert response.status == 404


class TestListFilesHandler:
    """Test /hf_downloader/files/<model_type> endpoint"""

    @pytest.mark.asyncio
    async def test_list_files_success(self, tmp_path):
        """Test successful file listing"""
        # Create test directory and files
        test_dir = tmp_path / "checkpoints"
        test_dir.mkdir()
        test_file = test_dir / "test_model.safetensors"
        test_file.write_bytes(b"test data")

        mock_request = Mock()
        mock_request.match_info = {"model_type": "checkpoint"}

        with patch("server_routes.get_model_dir", return_value=str(test_dir)):
            response = await list_files_handler(mock_request)
            assert response.status == 200

    @pytest.mark.asyncio
    async def test_list_files_nonexistent_dir(self):
        """Test listing files in non-existent directory"""
        mock_request = Mock()
        mock_request.match_info = {"model_type": "checkpoint"}

        with patch("server_routes.get_model_dir", return_value="/nonexistent/path"):
            response = await list_files_handler(mock_request)
            assert response.status == 200  # Returns empty list, not error


class TestDeleteFileHandler:
    """Test /hf_downloader/files DELETE endpoint"""

    @pytest.mark.asyncio
    async def test_delete_file_success(self, tmp_path):
        """Test successful file deletion"""
        # Create test file
        test_file = tmp_path / "test_model.safetensors"
        test_file.write_bytes(b"test data")

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"filepath": str(test_file)})

        with patch("server_routes.folder_paths") as mock_folder_paths:
            mock_folder_paths.models_dir = str(tmp_path)
            response = await delete_file_handler(mock_request)
            assert response.status == 200
            assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file_missing_filepath(self):
        """Test deletion with missing filepath"""
        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={})

        response = await delete_file_handler(mock_request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_delete_file_security_check(self, tmp_path):
        """Test security check prevents deletion outside models dir"""
        # Try to delete file outside models directory
        test_file = tmp_path / "outside" / "test.safetensors"
        test_file.parent.mkdir()
        test_file.write_bytes(b"test data")

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"filepath": str(test_file)})

        with patch("server_routes.folder_paths") as mock_folder_paths:
            mock_folder_paths.models_dir = str(tmp_path / "models")
            response = await delete_file_handler(mock_request)
            assert response.status == 403
            assert test_file.exists()  # File should not be deleted

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, tmp_path):
        """Test deletion of non-existent file"""
        nonexistent = tmp_path / "models" / "nonexistent.safetensors"

        mock_request = Mock()
        mock_request.json = AsyncMock(return_value={"filepath": str(nonexistent)})

        with patch("server_routes.folder_paths") as mock_folder_paths:
            mock_folder_paths.models_dir = str(tmp_path / "models")
            response = await delete_file_handler(mock_request)
            assert response.status == 404
