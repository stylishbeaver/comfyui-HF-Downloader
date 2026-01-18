"""
ComfyUI HuggingFace Downloader
Downloads and auto-merges split safetensor files from HuggingFace repositories
"""

from .server_routes import register_routes

# Extension metadata
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Register API routes on import
register_routes()
