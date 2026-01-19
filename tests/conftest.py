"""
Pytest configuration and fixtures
Sets up mocks for ComfyUI dependencies
"""

import sys
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock ComfyUI modules IMMEDIATELY before any imports
class MockFolderPaths:
    models_dir: ClassVar[str] = "/fake/models"
    folder_names_and_paths: ClassVar[dict] = {
        "checkpoints": (["/fake/models/checkpoints"], None),
        "loras": (["/fake/models/loras"], None),
    }


class MockPromptServer:
    def __init__(self):
        from aiohttp import web

        self.routes = web.RouteTableDef()


# Create mock prompt server instance
mock_prompt_server_instance = MockPromptServer()

# Create mock server module with PromptServer class
mock_server = Mock()
mock_server.PromptServer = Mock()
mock_server.PromptServer.instance = mock_prompt_server_instance

# Install mocks at module level BEFORE any test imports
sys.modules["folder_paths"] = MockFolderPaths()
sys.modules["server"] = mock_server
