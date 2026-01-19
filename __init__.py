"""
ComfyUI HuggingFace Downloader
Downloads and auto-merges split safetensor files from HuggingFace repositories
"""

# Extension metadata
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Import server_routes to register routes via decorators
# Only import when running in ComfyUI environment (not during tests)
try:
    from . import server_routes

    print("[HF Downloader] Extension loaded successfully")
except ImportError:
    pass  # Running in test environment or standalone
