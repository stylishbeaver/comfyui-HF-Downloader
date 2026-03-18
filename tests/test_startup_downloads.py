"""
Tests for startup_downloads split file expansion.
"""

import pytest

from startup_downloads import _expand_split_shards


class TestExpandSplitShards:
    """Test auto-detection and expansion of split shard filenames."""

    def test_expands_3_shards(self):
        """A filename matching the split pattern expands to all shards."""
        result = _expand_split_shards("model-00001-of-00003.safetensors")
        assert result == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

    def test_expands_from_middle_shard(self):
        """Any shard index (not just 00001) triggers expansion."""
        result = _expand_split_shards("model-00002-of-00003.safetensors")
        assert result == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

    def test_expands_large_shard_count(self):
        """Works for double-digit shard counts."""
        result = _expand_split_shards("weights-00001-of-00012.safetensors")
        assert len(result) == 12
        assert result[0] == "weights-00001-of-00012.safetensors"
        assert result[11] == "weights-00012-of-00012.safetensors"

    def test_single_file_unchanged(self):
        """A non-split filename returns a single-element list."""
        result = _expand_split_shards("model.safetensors")
        assert result == ["model.safetensors"]

    def test_non_safetensors_unchanged(self):
        """Non-safetensors files are never treated as splits."""
        result = _expand_split_shards("model.gguf")
        assert result == ["model.gguf"]

    def test_split_files_folder_name_not_confused(self):
        """The folder name 'split_files' in repo_path should NOT trigger expansion.

        This function only receives the filename part (after rsplit on '/'),
        so folder names are irrelevant — but verify the pattern is strict.
        """
        result = _expand_split_shards("z_image_turbo_bf16.safetensors")
        assert result == ["z_image_turbo_bf16.safetensors"]

    def test_preserves_base_name_with_hyphens(self):
        """Base names containing hyphens are preserved correctly."""
        result = _expand_split_shards("my-cool-model-00001-of-00002.safetensors")
        assert result == [
            "my-cool-model-00001-of-00002.safetensors",
            "my-cool-model-00002-of-00002.safetensors",
        ]

    def test_single_shard_of_one(self):
        """Edge case: 00001-of-00001 is technically a split pattern but returns one file."""
        result = _expand_split_shards("model-00001-of-00001.safetensors")
        assert result == ["model-00001-of-00001.safetensors"]
