# Captiv

Image captioning library using BLIP, BLIP-2, and JoyCaption models.

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or Poetry package manager
- For GPU acceleration (recommended):
  - CUDA-compatible GPU
  - CUDA Toolkit 11.7+ and cuDNN (for PyTorch GPU support)

### Basic Installation

```bash
# Using pip
pip install captiv

# Using Poetry
poetry install
```

### Virtual Environment (Recommended)

It's recommended to install Captiv in a virtual environment:

```bash
# Using venv
python -m venv captiv-env
source captiv-env/bin/activate  # On Windows: captiv-env\Scripts\activate
pip install captiv

# Using Poetry (automatically creates a virtual environment)
poetry new captiv-project
cd captiv-project
poetry add captiv
```

### JoyCaption Support

To use JoyCaption models, you need to install the `accelerate` package:

```bash
# Using pip
pip install captiv[joycaption]
# or
pip install accelerate

# Using Poetry
poetry install -E joycaption
# or
poetry add accelerate
```

### Development Installation

For development or contributing to Captiv:

```bash
# Clone the repository
git clone https://github.com/yourusername/captiv.git
cd captiv

# Complete setup (recommended)
make setup

# Or perform individual steps:
# Install dependencies
poetry install

# Set up pre-commit hooks
make setup-hooks
```

The `make setup` command performs a complete development environment setup:

1. Verifies Python 3.12+ is installed (errors if not)
2. Installs Poetry if not already installed
3. Installs all dependencies (Poetry automatically creates a virtual environment)
4. Sets up pre-commit hooks
5. Attempts to check hardware acceleration (CUDA/MPS) availability (continues even if this fails)
6. Runs linting checks and tests

The pre-commit hooks will automatically:

- Run code formatting (autoflake, isort, ruff format) when you commit
- Run linting checks and tests when you push

This ensures code quality standards are maintained throughout development.

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Generate a caption for an image
captiv caption generate path/to/image.jpg

# Generate a caption using a specific model
captiv caption generate path/to/image.jpg --model blip2

# Generate a caption using JoyCaption
captiv caption generate path/to/image.jpg --model joycaption
```

#### Batch Processing

```bash
# Generate captions for all images in a directory
captiv caption generate path/to/images/

# Clear all captions in a directory
captiv caption clear path/to/images/
```

#### Model Information

```bash
# List available models
captiv model list

# Show details for a specific model
captiv model show blip2

# List available caption modes
captiv caption list-modes

# List available model variants
captiv caption list-variants
```

### Graphical User Interface

The GUI requires Gradio 4.44.1, which is automatically installed with the package.

```bash
# Launch the GUI (localhost only)
captiv gui launch

# Launch the GUI with a public URL
captiv gui launch --share
```

#### GUI Server Configuration

You can configure the Gradio server settings:

```bash
# Set the host address (default: 127.0.0.1)
captiv config set gui.host "0.0.0.0"

# Set the server port (default: 7860)
captiv config set gui.port 8080

# Configure SSL for HTTPS
captiv config set gui.ssl_keyfile "/path/to/key.pem"
captiv config set gui.ssl_certfile "/path/to/cert.pem"
```

These settings allow you to:

- Change the host address (use "0.0.0.0" to listen on all network interfaces)
- Change the port number
- Enable HTTPS by providing SSL certificate and key files

### Python API

#### Basic Usage

```python
from captiv.models import BlipModel, Blip2Model, JoyCaptionModel
from PIL import Image

# Using BLIP
model = BlipModel()
image = Image.open("path/to/image.jpg")
caption = model.caption_image(image)
print(caption)

# Using BLIP-2
model = Blip2Model()
caption = model.caption_image("path/to/image.jpg")
print(caption)

# Using JoyCaption (requires accelerate package)
model = JoyCaptionModel()
caption = model.caption_image(
    "path/to/image.jpg",
    mode="descriptive_formal"
)
print(caption)
```

#### Batch Processing with Error Handling

```python
from captiv.models import Blip2Model
import os
from pathlib import Path

model = Blip2Model()
image_dir = Path("path/to/images")

for image_file in image_dir.glob("*.jpg"):
    try:
        caption = model.caption_image(str(image_file))
        # Save caption to a sidecar text file
        with open(image_file.with_suffix(".txt"), "w") as f:
            f.write(caption)
        print(f"Captioned: {image_file.name}")
    except Exception as e:
        print(f"Error captioning {image_file.name}: {str(e)}")
```

## Models

### BLIP

BLIP (Bootstrapping Language-Image Pre-training) is a vision-language model that can generate captions for images. It offers a good balance between speed and quality.

**Key Features:**

- Fast inference
- Moderate memory requirements
- Good for general-purpose captioning

### BLIP-2

BLIP-2 is an improved version of BLIP with better performance and more features. It uses a more advanced architecture that combines vision encoders with large language models.

**Key Features:**

- Higher quality captions than BLIP
- More detailed descriptions
- Better understanding of complex scenes
- Higher memory requirements

### JoyCaption

JoyCaption is a specialized image captioning model with various modes and styles. It requires the `accelerate` package for optimal performance.

**Key Features:**

- Multiple captioning styles
- Highly detailed descriptions
- Support for specialized use cases (e.g., generating prompts for image generation)
- Higher computational requirements

#### JoyCaption Modes

- `descriptive_formal`: Generate formal, detailed descriptions
- `descriptive_casual`: Write descriptive captions in a casual tone
- `straightforward`: Write straightforward captions with concrete details
- `stable_diffusion`: Generate Stable Diffusion prompts
- `midjourney`: Generate MidJourney prompts
- And many more...

## Configuration

You can configure default settings in the configuration file:

```bash
# Get current configuration
captiv config list

# Set default model
captiv config set model.default_model blip2

# Set default JoyCaption variant
captiv config set model.joycaption_variant "fancyfeast/llama-joycaption-beta-one-hf-llava"
```

## Troubleshooting

### Installation Issues

#### Missing Dependencies

**Problem:** Error about missing dependencies when installing or running Captiv.

**Solution:**

```bash
# Install all dependencies
pip install -r requirements.txt

# For JoyCaption-specific issues
pip install accelerate
```

#### Poetry Installation Errors

**Problem:** Poetry fails to resolve dependencies.

**Solution:**

```bash
# Update Poetry
poetry self update

# Clear Poetry cache
poetry cache clear --all .

# Try installation again
poetry install
```

### GPU/CUDA Issues

#### CUDA Not Found

**Problem:** "CUDA not available" or "No CUDA GPUs are available" errors.

**Solution:**

1. Verify CUDA installation:

   ```bash
   nvidia-smi
   ```

2. Ensure PyTorch is installed with CUDA support:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Reinstall PyTorch with CUDA support:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

#### Out of Memory Errors

**Problem:** CUDA out of memory errors when running models.

**Solution:**

1. Use a smaller model variant:

   ```bash
   captiv caption generate image.jpg --model blip --variant base
   ```

2. Reduce batch size or image resolution in your code
3. Close other GPU-intensive applications

### Model Download Issues

#### Slow or Failed Downloads

**Problem:** Model downloads are slow, timeout, or fail.

**Solution:**

1. Check your internet connection
2. Use a VPN if your location restricts access to model repositories
3. Manually download models from Hugging Face and place them in the cache directory:
   - Linux/Mac: `~/.cache/huggingface/hub/`
   - Windows: `C:\Users\<username>\.cache\huggingface\hub\`

#### Disk Space Issues

**Problem:** Not enough disk space for model downloads.

**Solution:**

1. Free up disk space
2. Use smaller model variants
3. Set a custom cache directory with more space:

   ```bash
   export TRANSFORMERS_CACHE="/path/with/more/space"
   ```

### GUI Issues

#### Gradio Installation Problems

**Problem:** Errors related to Gradio when launching the GUI.

**Solution:**

```bash
# Install the exact required version
pip install gradio==4.44.1

# If conflicts occur, try in a fresh virtual environment
python -m venv captiv-gui-env
source captiv-gui-env/bin/activate
pip install captiv
pip install gradio==4.44.1
```

#### GUI Access Issues

**Problem:** Cannot access the GUI from another device.

**Solution:**

1. Configure the GUI to listen on all interfaces:

   ```bash
   captiv config set gui.host "0.0.0.0"
   captiv gui launch
   ```

2. Ensure your firewall allows connections to the port (default: 7860)
3. Use the `--share` option for a public URL:

   ```bash
   captiv gui launch --share
   ```

### Permission Issues

**Problem:** Permission denied errors when reading images or writing captions.

**Solution:**

1. Check file and directory permissions
2. Run the command with appropriate permissions
3. Ensure the application has write access to the directory for sidecar caption files

## Dependencies

- Python 3.12+
- PyTorch: Deep learning framework
- Transformers: Provides model implementations
- Pillow: Image processing
- Typer: Command-line interface
- Gradio 4.44.1: Required for GUI
- Accelerate: Optional, required for JoyCaption

## License

[License information here]
