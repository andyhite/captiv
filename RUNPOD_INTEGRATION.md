# RunPod Integration for Captiv

This document describes how to use Captiv with RunPod for remote GPU inference, allowing you to run JoyCaption models on cloud GPUs without local hardware requirements.

## Overview

The RunPod integration allows you to:

- Create and manage RunPod instances remotely
- Automatically deploy Captiv with JoyCaption models
- Generate captions using remote GPU inference
- Seamlessly fall back to local inference if needed

## Setup

### 1. Get RunPod API Key

1. Sign up at [RunPod.io](https://runpod.io)
2. Go to Settings → API Keys
3. Create a new API key
4. Copy the API key for configuration

### 2. Configure Captiv

Set your RunPod API key:

```bash
captiv config set runpod api_key YOUR_API_KEY_HERE
```

Optional configuration:

```bash
# Set default GPU type
captiv config set runpod default_gpu_type "NVIDIA RTX A4000"

# Enable auto-creation of pods
captiv config set runpod auto_create true

# Enable auto-termination when done
captiv config set runpod auto_terminate true

# Set startup timeout (seconds)
captiv config set runpod startup_timeout 600
```

## Usage

### Using RunPod with Any Model

Use the `--runpod` flag with any model for automatic RunPod integration:

```bash
# Generate captions using RunPod (will auto-create pod if needed)
captiv caption generate --runpod --model joycaption /path/to/images/

# Generate with specific mode
captiv caption generate --runpod --model blip2 --mode default /path/to/image.jpg

# Use with any supported model
captiv caption generate --runpod --model kosmos /path/to/image.jpg
```

### Using RunPod with GUI

Launch the GUI with RunPod support:

```bash
# Launch GUI with RunPod support
captiv gui launch --runpod

# Launch GUI with RunPod and public sharing
captiv gui launch --runpod --share
```

### Manual Pod Management

Create a RunPod instance:

```bash
# Create a new pod
captiv runpod create --name my-captiv-pod --gpu-type "NVIDIA RTX A4000"

# Create and wait for it to be ready
captiv runpod create --name my-captiv-pod --wait
```

Check pod status:

```bash
captiv runpod status POD_ID
```

Stop a pod (can be restarted):

```bash
captiv runpod stop POD_ID
```

Terminate a pod (permanent deletion):

```bash
captiv runpod terminate POD_ID
```

List pods:

```bash
captiv runpod list
```

## Configuration Options

All RunPod settings can be configured via the CLI:

| Setting               | Description                   | Default                  |
| --------------------- | ----------------------------- | ------------------------ |
| `api_key`             | RunPod API key                | None (required)          |
| `template_id`         | RunPod template ID            | None (uses Docker image) |
| `default_gpu_type`    | Default GPU type for pods     | "NVIDIA RTX A4000"       |
| `auto_create`         | Auto-create pods when needed  | true                     |
| `auto_terminate`      | Auto-terminate pods when done | true                     |
| `startup_timeout`     | Pod startup timeout (seconds) | 600                      |
| `container_disk_size` | Container disk size (GB)      | 50                       |
| `volume_disk_size`    | Volume disk size (GB)         | 50                       |
| `request_timeout`     | API request timeout (seconds) | 120                      |

## How It Works

### Automatic Mode

When using the `--runpod` flag:

1. Captiv checks if a RunPod instance is available
2. If not, creates a new pod with the Captiv Docker image
3. Waits for the pod to be ready and healthy
4. Sends images to the pod's API endpoint for processing
5. Returns generated captions
6. Optionally terminates the pod when done

The same model type is used both locally and remotely, ensuring consistent results.

The same model type is used both locally and remotely, ensuring consistent results.

### Fallback Behavior

If RunPod fails or is unavailable:

- Automatically falls back to local inference
- Provides clear error messages and troubleshooting tips
- Continues operation without interruption

### Docker Image

The Captiv Docker image includes:

- Pre-installed JoyCaption models
- FastAPI server for remote inference
- Health check endpoints
- Automatic model loading on startup

## API Endpoints

When a RunPod instance is running, it exposes these endpoints:

- `GET /health` - Health check
- `POST /api/caption` - Generate captions
- `GET /api/models` - List available models
- `GET /api/modes/{model_name}` - Get model modes

## Troubleshooting

### Common Issues

**Pod creation fails:**

- Check your RunPod API key
- Verify you have sufficient credits
- Try a different GPU type

**Pod not responding:**

- Wait for startup to complete (can take 5-10 minutes)
- Check pod status with `captiv runpod status POD_ID`
- Verify health check passes

**Caption generation fails:**

- Ensure pod is healthy
- Check image format is supported
- Verify network connectivity

### Debug Commands

```bash
# Check configuration
captiv config list

# Test pod health
captiv runpod status POD_ID

# View pod logs (if accessible)
# Log into RunPod web interface for detailed logs
```

## Cost Optimization

### Tips for Reducing Costs

1. **Use auto-terminate**: Automatically terminate pods when done
2. **Choose appropriate GPU**: Use smaller GPUs for testing
3. **Batch processing**: Process multiple images in one session
4. **Monitor usage**: Check RunPod dashboard regularly

### GPU Recommendations

| Use Case   | Recommended GPU | Notes                            |
| ---------- | --------------- | -------------------------------- |
| Testing    | RTX A4000       | Good balance of cost/performance |
| Production | RTX A5000/A6000 | Faster inference, higher cost    |
| Budget     | RTX 3070        | Lower cost, adequate performance |

## Security

- API keys are stored locally in Captiv configuration
- All communication uses HTTPS
- No image data is permanently stored on RunPod instances
- Pods are isolated and automatically cleaned up

## Limitations

- Requires internet connection
- Subject to RunPod availability and pricing
- Initial pod startup takes 5-10 minutes
- Model-specific features may vary between local and remote inference

## Support

For issues with:

- **Captiv integration**: Create an issue on the Captiv repository
- **RunPod platform**: Contact RunPod support
- **Billing/credits**: Check RunPod dashboard and billing section
