# Captiv Image Captioning Application

This container provides both CLI and GUI interfaces for image captioning using various AI models.

## Quick Start

### CLI Usage

```bash
# Show help
captiv --help

# List available models
captiv model list

# Generate captions for images
captiv caption generate /path/to/images

# Launch GUI from CLI
captiv gui launch
```

### GUI Usage

- Access the web interface at: <http://localhost:7860>
- Upload images and select models for captioning
- Batch process multiple images
- Export captions in various formats

### Jupyter Lab

- Access Jupyter Lab at: <http://localhost:8888/lab>
- Explore the codebase and experiment with models
- Create custom captioning workflows

## Directory Structure

- `/workspace/captiv/` - Application source code
- `/workspace/models/` - Downloaded AI models
- `/workspace/outputs/` - Generated captions and results
- `/workspace/logs/` - Application and service logs

## Environment Variables

- `CAPTIV_HOST` - GUI host (default: 0.0.0.0)
- `CAPTIV_PORT` - GUI port (default: 7860)
- `CAPTIV_SHARE` - Enable public sharing (true/false)
- `CAPTIV_CONFIG_PATH` - Configuration directory

## Documentation

For detailed documentation, visit: <https://github.com/andyhite/captiv>

## Support

- Report issues: <https://github.com/andyhite/captiv/issues>
- Discussions: <https://github.com/andyhite/captiv/discussions>
