# ComfyUI HuggingFace Downloader

A ComfyUI custom extension for downloading and automatically merging split safetensor files from HuggingFace repositories. Also supports GGUF, PyTorch (.pt/.pth) model formats.

## Features

- 🚀 **Lightning-fast downloads** using HuggingFace CLI (2-3 GB/s)
- 🔄 **Automatic split detection** - finds and merges model shards automatically
- 📁 **Smart organization** - detects model types and saves to appropriate directories
- 🎯 **Intelligent naming** - suggests output names from repo/folder structure
- 📊 **Progress tracking** - real-time download and merge progress
- 🔐 **Token support** - uses `HF_TOKEN` environment variable for authentication
- 🎨 **Multiple formats** - supports safetensors, GGUF, PyTorch (.pt/.pth) models

## Problem This Solves

Many HuggingFace models are split into multiple files like:
```
repo/
├── model_a/
│   ├── model-00001-of-00003.safetensors
│   ├── model-00002-of-00003.safetensors
│   └── model-00003-of-00003.safetensors
```

This extension automatically:
1. Detects all split files in a repository
2. Downloads them using the fast HF CLI
3. Merges them into a single usable `.safetensors` file
4. Saves to the correct ComfyUI models directory

## Installation

### Prerequisites

- ComfyUI installed
- HuggingFace CLI installed (see below)
- Python packages: `huggingface_hub`, `safetensors` (usually already in ComfyUI)

### Install HuggingFace CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Install Extension

1. Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-HF-Downloader.git
```

2. Restart ComfyUI

3. (Optional) Set your HuggingFace token as an environment variable:

```bash
export HF_TOKEN=hf_...
```

Or in RunPod/Docker environments, add it to your environment variables.

## Usage

### Basic Workflow

1. **Open the HF Downloader**
   - Click the "HF Downloader" button in the ComfyUI top menu

2. **Enter Repository URL**
   - Paste a HuggingFace repo URL (e.g., `username/model-name`)
   - Click "Scan Repo"

3. **Select Models**
   - The extension will list all detected models
   - Shows whether each model is split or single file
   - Edit the output name if desired
   - Select the appropriate model type (Checkpoint, LoRA, VAE, etc.)

4. **Download**
   - Click "Download" for any model
   - Watch real-time progress (download → merge → save)
   - Files are saved to the appropriate ComfyUI models directory

### Example Repositories

Try these HuggingFace repos with split files:

- `Comfy-Org/z_image_turbo` - Has split files in subfolders
- Large transformer models often have splits

### Model Types

The extension supports all ComfyUI model types:

- **Checkpoint** → `ComfyUI/models/checkpoints/`
- **LoRA** → `ComfyUI/models/loras/`
- **VAE** → `ComfyUI/models/vae/`
- **Upscale** → `ComfyUI/models/upscale_models/`
- **Embedding** → `ComfyUI/models/embeddings/`
- **CLIP** → `ComfyUI/models/clip/`
- **ControlNet** → `ComfyUI/models/controlnet/`
- **Diffusion Model** → `ComfyUI/models/diffusion_models/`
- **Text Encoder** → `ComfyUI/models/text_encoders/`

## Configuration

### HuggingFace Token

For private repos or faster download speeds, set your HF token:

```bash
# Linux/Mac
export HF_TOKEN=hf_your_token_here

# Or in your shell profile
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc

# Windows (PowerShell)
$env:HF_TOKEN="hf_your_token_here"

# Docker/RunPod
# Add HF_TOKEN as an environment variable in your container config
```

Get your token from: https://huggingface.co/settings/tokens

## Technical Details

### How Split Detection Works

The extension scans for patterns like:
- `model-00001-of-00003.safetensors`
- `model-00002-of-00003.safetensors`
- etc.

It groups files by subfolder and detects the split pattern using regex.

### How Merging Works

Uses the official `safetensors` library:

```python
from safetensors.torch import load_file, save_file

# Load all shards
tensors = {}
for shard in sorted(glob.glob("model-*.safetensors")):
    tensors.update(load_file(shard))

# Save as single file
save_file(tensors, "model-merged.safetensors")
```

### Download Speed

Uses HuggingFace CLI which provides:
- Parallel chunk downloads
- Optimized CDN routing
- ~2-3 GB/s speeds (vs ~300 MB/s with curl)
- Automatic caching (files stored in `~/.cache/huggingface/`)

## API Endpoints

The extension exposes these API endpoints:

### `POST /hf_downloader/scan`
Scan a repository for models

```json
{
  "repo_id": "username/model"
}
```

### `POST /hf_downloader/download`
Start downloading a model

```json
{
  "repo_id": "username/model",
  "model_path": "subfolder",
  "files": ["file1.safetensors", "file2.safetensors"],
  "output_name": "model_name",
  "model_type": "checkpoint"
}
```

### `GET /hf_downloader/progress/{task_id}`
Get download progress

## Troubleshooting

### "HF CLI not found"

Install the HuggingFace CLI:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

### "Authentication required"

Set your HF token:
```bash
export HF_TOKEN=hf_your_token_here
```

### Downloads are slow

- Make sure HF CLI is installed (not falling back to curl)
- Check your internet connection
- Verify `HF_TOKEN` is set for better routing

### Files not appearing in ComfyUI

- Restart ComfyUI after downloads complete
- Check the console for the actual save path
- Verify you selected the correct model type

## Development

### Project Structure

```
ComfyUI-HF-Downloader/
├── __init__.py              # Extension registration
├── hf_downloader.py         # Core download/merge logic
├── server_routes.py         # API endpoints
└── web/
    └── js/
        └── hf_downloader.js # Frontend UI
```

### Running Tests

```bash
# Test scanning a repo
python -c "from hf_downloader import HFDownloader; d = HFDownloader(); print(d.scan_repo('Comfy-Org/z_image_turbo'))"
```

## Credits

- Built for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Uses [HuggingFace Hub](https://github.com/huggingface/huggingface_hub)
- Uses [safetensors](https://github.com/huggingface/safetensors)

## License

MIT

## Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- Test with real HuggingFace repos
- Update README if adding features
