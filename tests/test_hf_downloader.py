"""
Comprehensive tests for HFDownloader class
Tests cover split detection, precision extraction, name suggestion, and merge operations
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hf_downloader import HFDownloader


class TestHFDownloader:
    """Test suite for HFDownloader class"""

    @pytest.fixture
    def downloader(self):
        """Create a HFDownloader instance for testing"""
        with patch.dict("os.environ", {"HF_TOKEN": "test_token"}):
            return HFDownloader()

    @pytest.fixture
    def downloader_no_token(self):
        """Create a HFDownloader instance without token"""
        with patch.dict("os.environ", {}, clear=True):
            return HFDownloader()

    def test_init_with_token(self, downloader):
        """Test initialization with HF token"""
        assert downloader.hf_token == "test_token"
        assert downloader.api is not None

    def test_init_without_token(self, downloader_no_token):
        """Test initialization without HF token"""
        assert downloader_no_token.hf_token is None
        assert downloader_no_token.api is not None


class TestPrecisionExtraction:
    """Test precision extraction from filenames"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("model_fp16.safetensors", "fp16"),
            ("model-fp32.safetensors", "fp32"),
            ("model_bf16.safetensors", "bf16"),
            ("model-fp8.safetensors", "fp8"),
            ("model_int8.safetensors", "int8"),
            ("model-int4.safetensors", "int4"),
            ("model_bnb-4bit.safetensors", "bnb-4bit"),
            ("model-bnb_8bit.safetensors", "bnb-8bit"),
            ("model_fp8_e4m3fn.safetensors", "fp8-e4m3fn"),
            ("model_nvfp4.safetensors", "nvfp4"),
            ("model.safetensors", None),
            ("model_v2.safetensors", None),
        ],
    )
    def test_extract_precision(self, downloader, filename, expected):
        """Test precision extraction from various filename patterns"""
        result = downloader._extract_precision(filename)
        assert result == expected

    def test_precision_case_insensitive(self, downloader):
        """Test that precision extraction is case-insensitive"""
        assert downloader._extract_precision("model_FP16.safetensors") == "fp16"
        assert downloader._extract_precision("model-BF16.safetensors") == "bf16"


class TestSplitDetection:
    """Test detection of split model files"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    def test_detect_split_files(self, downloader):
        """Test detection of split pattern in filenames"""
        files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        result = downloader._detect_splits(files)
        assert result is not None
        assert result["total"] == 3
        assert "pattern" in result

    def test_detect_no_splits(self, downloader):
        """Test that non-split files return None"""
        files = ["model.safetensors", "vae.safetensors"]
        result = downloader._detect_splits(files)
        assert result is None

    def test_detect_single_file(self, downloader):
        """Test single file returns None"""
        files = ["model.safetensors"]
        result = downloader._detect_splits(files)
        assert result is None

    def test_detect_mixed_files(self, downloader):
        """Test detection with mixed split and non-split files"""
        files = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "vae.safetensors",
        ]
        # Should detect the split pattern from first file
        result = downloader._detect_splits(files)
        assert result is not None
        assert result["total"] == 2


class TestNameSuggestion:
    """Test filename suggestion logic"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    def test_suggest_name_root_single_file(self, downloader):
        """Test name suggestion for single file in root"""
        files = ["model_fp16.safetensors"]
        result = downloader._suggest_name("user/repo", "root", files, None)
        assert result == "model_fp16"

    def test_suggest_name_split_files(self, downloader):
        """Test name suggestion for split files"""
        files = [
            "base_model-00001-of-00003.safetensors",
            "base_model-00002-of-00003.safetensors",
            "base_model-00003-of-00003.safetensors",
        ]
        split_info = {"total": 3, "pattern": r"-(\d+)-of-(\d+)\.safetensors$"}
        result = downloader._suggest_name("user/repo", "root", files, split_info)
        assert result == "base_model"

    def test_suggest_name_folder(self, downloader):
        """Test name suggestion for files in a folder"""
        files = ["model.safetensors"]
        with patch.object(downloader, "_get_name_from_config", return_value=None):
            result = downloader._suggest_name("user/repo", "models/v1", files, None)
            assert result == "v1"

    def test_suggest_name_fallback_to_repo(self, downloader):
        """Test fallback to repo name"""
        files = ["model.safetensors"]
        result = downloader._suggest_name("user/awesome-model", "root", files, None)
        # For root with single file, should use filename
        assert result == "model"

    def test_suggest_name_with_config(self, downloader):
        """Test name suggestion using config.json"""
        files = ["model.safetensors"]
        with patch.object(downloader, "_get_name_from_config", return_value="custom_name"):
            result = downloader._suggest_name("user/repo", "models/v1", files, None)
            assert result == "custom_name"


class TestConfigParsing:
    """Test config.json parsing for model names"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    def test_get_name_from_config_name_or_path(self, downloader):
        """Test extracting name from _name_or_path field"""
        import json
        from io import StringIO

        mock_config = {"_name_or_path": "google/t5-v1_1-xxl", "model_type": "t5"}
        mock_file_content = StringIO(json.dumps(mock_config))

        with (
            patch("hf_downloader.hf_hub_download", return_value="/tmp/config.json"),
            patch("pathlib.Path.open", return_value=mock_file_content),
        ):
            result = downloader._get_name_from_config("user/repo", "folder")
            assert result == "t5-v1_1-xxl"

    def test_get_name_from_config_model_type_fallback(self, downloader):
        """Test fallback to model_type when _name_or_path is absent"""
        import json
        from io import StringIO

        mock_config = {"model_type": "bert"}
        mock_file_content = StringIO(json.dumps(mock_config))

        with (
            patch("hf_downloader.hf_hub_download", return_value="/tmp/config.json"),
            patch("pathlib.Path.open", return_value=mock_file_content),
        ):
            result = downloader._get_name_from_config("user/repo", "folder")
            assert result == "bert"

    def test_get_name_from_config_not_found(self, downloader):
        """Test when config.json cannot be found"""
        with patch("hf_downloader.hf_hub_download", side_effect=Exception("File not found")):
            result = downloader._get_name_from_config("user/repo", "folder")
            assert result is None


class TestScanRepo:
    """Test repository scanning functionality"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    def test_scan_repo_basic(self, downloader):
        """Test basic repo scanning with mock data"""
        # Create mock file info objects
        mock_files = [
            Mock(rfilename="model.safetensors", size=1000000),
            Mock(rfilename="vae/vae_fp16.safetensors", size=500000),
        ]

        # Mock the api.list_files_info method
        downloader.api.list_files_info = Mock(return_value=mock_files)

        result = downloader.scan_repo("user/test-repo")

        assert len(result) == 2
        # Check root model
        assert any(item["path"] == "root" for item in result)
        # Check vae folder
        assert any(item["path"] == "vae" for item in result)

    def test_scan_repo_split_files(self, downloader):
        """Test repo scanning with split files"""
        mock_files = [
            Mock(rfilename="model/file-00001-of-00003.safetensors", size=1000000),
            Mock(rfilename="model/file-00002-of-00003.safetensors", size=1000000),
            Mock(rfilename="model/file-00003-of-00003.safetensors", size=1000000),
        ]

        downloader.api.list_files_info = Mock(return_value=mock_files)

        result = downloader.scan_repo("user/test-repo")

        assert len(result) == 1
        assert result[0]["is_split"] is True
        assert result[0]["file_count"] == 3
        assert result[0]["total_size"] == 3000000

    def test_scan_repo_multiple_root_files(self, downloader):
        """Test that multiple root files are treated as separate entries"""
        mock_files = [
            Mock(rfilename="model1.safetensors", size=1000000),
            Mock(rfilename="model2.safetensors", size=2000000),
        ]

        downloader.api.list_files_info = Mock(return_value=mock_files)

        result = downloader.scan_repo("user/test-repo")

        # Should have 2 separate entries for root files
        assert len(result) == 2
        assert all(item["path"] == "root" for item in result)
        assert all(item["file_count"] == 1 for item in result)

    def test_scan_repo_subfolder_variants_grouped(self, downloader):
        """Test that non-split variants in the same folder are grouped by base name"""
        mock_files = [
            Mock(rfilename="models/qwen_image_bf16.safetensors", size=1000000),
            Mock(rfilename="models/qwen_image_fp8_e4m3fn.safetensors", size=2000000),
            Mock(rfilename="models/other.safetensors", size=3000000),
        ]

        downloader.api.list_files_info = Mock(return_value=mock_files)

        result = downloader.scan_repo("user/test-repo")

        assert len(result) == 2
        qwen_group = next(item for item in result if item.get("base_name") == "qwen_image")
        assert qwen_group["file_count"] == 2
        assert qwen_group["is_split"] is False
        assert qwen_group["suggested_name"] == "qwen_image"
        assert {f["name"] for f in qwen_group["files"]} == {
            "qwen_image_bf16.safetensors",
            "qwen_image_fp8_e4m3fn.safetensors",
        }
        other_group = next(item for item in result if item.get("base_name") == "other")
        assert other_group["file_count"] == 1
        assert other_group["total_size"] == 3000000

    def test_scan_repo_error_handling(self, downloader):
        """Test error handling in repo scanning"""
        downloader.api.list_files_info = Mock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            downloader.scan_repo("user/test-repo")

    def test_scan_repo_gguf_variants(self, downloader):
        """Test repo scanning with GGUF quant variants"""
        mock_files = [
            Mock(rfilename="llm/model.Q4_K_M.gguf", size=1000000),
            Mock(rfilename="llm/model.Q5_K_M.gguf", size=2000000),
            Mock(rfilename="llm/other.gguf", size=500000),
        ]

        downloader.api.list_files_info = Mock(return_value=mock_files)

        result = downloader.scan_repo("user/test-repo")
        gguf_entries = [item for item in result if item.get("file_type") == "gguf"]

        assert len(gguf_entries) == 2
        model_entry = next(item for item in gguf_entries if item.get("base_name") == "model")
        assert model_entry["file_count"] == 2
        assert set(model_entry["quant_options"]) == {"Q4_K_M", "Q5_K_M"}


class TestGGUFQuantExtraction:
    """Test GGUF quant extraction from filenames"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("model.Q4_K_M.gguf", "Q4_K_M"),
            ("model-q8_0.gguf", "Q8_0"),
            ("model.F16.gguf", "F16"),
            ("model-bf16.gguf", "BF16"),
            ("model.gguf", None),
            ("model_qwen.gguf", None),
        ],
    )
    def test_extract_gguf_quant(self, downloader, filename, expected):
        """Test quant extraction for common GGUF patterns"""
        assert downloader._extract_gguf_quant(filename) == expected


class TestHFCLIOperations:
    """Test HuggingFace CLI operations"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    def test_check_hf_cli_installed(self, downloader):
        """Test HF CLI availability check when installed"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = downloader._check_hf_cli()
            assert result is True

    def test_check_hf_cli_not_installed(self, downloader):
        """Test HF CLI availability check when not installed"""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = downloader._check_hf_cli()
            assert result is False

    def test_hf_login_success(self, downloader):
        """Test successful HF login"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            downloader._hf_login()  # Should not raise

    def test_hf_login_failure_nonfatal(self, downloader):
        """Test that HF login failure is non-fatal"""
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "hf", stderr="Error"),
        ):
            # Should not raise, just log warning
            downloader._hf_login()

    def test_download_file_success(self, downloader):
        """Test successful file download"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="/cache/path/model.safetensors")
            result = downloader._download_file("user/repo", "model.safetensors")
            assert result == "/cache/path/model.safetensors"

    def test_download_file_failure(self, downloader):
        """Test file download failure"""
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "hf", stderr="Download error"),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            downloader._download_file("user/repo", "model.safetensors")


class TestDownloadAndMerge:
    """Test download and merge operations"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_download_and_merge_no_cli(self, downloader, temp_dir):
        """Test that download fails gracefully when HF CLI is not available"""
        with (
            patch.object(downloader, "_check_hf_cli", return_value=False),
            pytest.raises(RuntimeError, match="HF CLI not found"),
        ):
            downloader.download_and_merge(
                repo_id="user/repo",
                folder_path="root",
                files=["model.safetensors"],
                output_dir=temp_dir,
                output_name="test_model",
            )

    def test_download_single_file(self, downloader, temp_dir):
        """Test downloading a single file (copy operation)"""
        # Create a fake cached file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            cached_path = f.name
            f.write(b"fake model data")

        try:
            with (
                patch.object(downloader, "_check_hf_cli", return_value=True),
                patch.object(downloader, "_hf_login"),
                patch.object(downloader, "_download_file", return_value=cached_path),
            ):
                result = downloader.download_and_merge(
                    repo_id="user/repo",
                    folder_path="root",
                    files=["model.safetensors"],
                    output_dir=temp_dir,
                    output_name="test_model",
                )

                assert result == str(Path(temp_dir) / "test_model.safetensors")
                result_path = Path(result)
                assert result_path.exists()
                if result_path.is_symlink():
                    assert result_path.resolve() == Path(cached_path).resolve()
                else:
                    assert result_path.read_bytes() == Path(cached_path).read_bytes()
        finally:
            Path(cached_path).unlink(missing_ok=True)

    def test_download_with_progress_callback(self, downloader, temp_dir):
        """Test that progress callbacks are called correctly"""
        # Create a fake cached file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            cached_path = f.name
            f.write(b"fake model data")

        progress_calls = []

        def progress_callback(stage, current, total, message):
            progress_calls.append((stage, current, total, message))

        try:
            with (
                patch.object(downloader, "_check_hf_cli", return_value=True),
                patch.object(downloader, "_hf_login"),
                patch.object(downloader, "_download_file", return_value=cached_path),
            ):
                downloader.download_and_merge(
                    repo_id="user/repo",
                    folder_path="root",
                    files=["model.safetensors"],
                    output_dir=temp_dir,
                    output_name="test_model",
                    progress_callback=progress_callback,
                )

                # Verify progress callbacks were called
                assert len(progress_calls) > 0
                # Check for expected stages
                stages = {call[0] for call in progress_calls}
                assert "download" in stages
                assert "copy" in stages
        finally:
            Path(cached_path).unlink(missing_ok=True)


class TestMergeFiles:
    """Test file merging functionality"""

    @pytest.fixture
    def downloader(self):
        return HFDownloader()

    def test_merge_files_basic(self, downloader):
        """Test basic merge operation with mock safetensors"""
        import torch
        from safetensors.torch import save_file

        # Create temporary shard files
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.NamedTemporaryFile(suffix=".safetensors", dir=tmpdir, delete=False) as f1,
            tempfile.NamedTemporaryFile(suffix=".safetensors", dir=tmpdir, delete=False) as f2,
        ):
            # Create mock tensor data
            shard1 = {"layer1.weight": torch.randn(10, 10)}
            shard2 = {"layer2.weight": torch.randn(10, 10)}

            save_file(shard1, f1.name)
            save_file(shard2, f2.name)

            output_path = str(Path(tmpdir) / "merged.safetensors")

            # Merge the files
            downloader._merge_files([f1.name, f2.name], output_path)

            # Verify merged file exists and contains all tensors
            assert Path(output_path).exists()

            from safetensors.torch import load_file

            merged = load_file(output_path)
            assert "layer1.weight" in merged
            assert "layer2.weight" in merged

    def test_merge_files_with_progress(self, downloader):
        """Test merge operation with progress callback"""
        import torch
        from safetensors.torch import save_file

        progress_calls = []

        def progress_callback(stage, current, total, message):
            progress_calls.append((stage, current, total, message))

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.NamedTemporaryFile(suffix=".safetensors", dir=tmpdir, delete=False) as f1,
        ):
            shard1 = {"layer1.weight": torch.randn(10, 10)}
            save_file(shard1, f1.name)

            output_path = str(Path(tmpdir) / "merged.safetensors")
            downloader._merge_files([f1.name], output_path, progress_callback)

            assert len(progress_calls) > 0
            assert all(call[0] == "merge" for call in progress_calls)
