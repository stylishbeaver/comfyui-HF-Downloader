"""
ComfyUI HuggingFace Downloader
Downloads and auto-merges split safetensor files from HuggingFace repositories
"""

# Import server_routes to register routes via decorators
from . import server_routes

# Extension metadata
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[HF Downloader] Extension loaded successfully")
