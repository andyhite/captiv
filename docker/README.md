# Docker Configuration Files

This directory contains configuration files that are copied into the Docker image during the build process, following RunPod best practices for workspace volume mounting.

## Directory Structure

```txt
docker/
├── README.md                    # This file
└── config/                      # Configuration files copied to container
    ├── nginx-default.conf       # NGINX proxy configuration
    ├── workspace-readme.md      # README for /workspace directory
    ├── runpod-welcome.txt       # Welcome message shown on login
    └── health-check.json        # Health check endpoint data
```

## RunPod Volume Mounting Pattern

RunPod mounts a persistent volume at `/workspace` when the container starts. This means:

1. **Anything placed in `/workspace` during Docker build is overwritten**
2. **Template files must be stored elsewhere and synced by the entrypoint**
3. **The application itself should be installed globally, not in `/workspace`**

## Configuration Files

### `config/nginx-default.conf`

- **Purpose**: NGINX reverse proxy configuration
- **Build Destination**: `/etc/nginx/sites-available/default`
- **Function**: Routes traffic to Captiv GUI (port 7860) and Jupyter Lab (port 8888)

### `config/workspace-readme.md`

- **Purpose**: User documentation for the workspace
- **Build Destination**: `/opt/workspace-template/README.md`
- **Runtime Destination**: `/workspace/README.md` (synced by entrypoint)
- **Function**: Provides usage instructions and directory structure info

### `config/runpod-welcome.txt`

- **Purpose**: Welcome message displayed on container startup
- **Build Destination**: `/etc/runpod.txt`
- **Function**: Shows available commands and endpoints to users

### `config/health-check.json`

- **Purpose**: Health check endpoint data template
- **Build Destination**: `/opt/workspace-template/captiv/health.json`
- **Runtime Destination**: `/workspace/captiv/health.json` (synced by entrypoint)
- **Function**: Provides service status information for monitoring

## Docker Build Process

The Dockerfile follows this pattern:

```dockerfile
# Install Captiv application globally at /captiv
WORKDIR /captiv
COPY src/ /captiv/src/
RUN pip install -e .

# Copy config files to template location (NOT /workspace)
COPY docker/config/ /opt/captiv-config/

# Create workspace template (will be synced to /workspace at runtime)
RUN mkdir -p /opt/workspace-template/{models,outputs,logs,captiv} && \
    cp /opt/captiv-config/workspace-readme.md /opt/workspace-template/README.md && \
    cp /opt/captiv-config/health-check.json /opt/workspace-template/captiv/health.json
```

## Entrypoint Sync Process

The entrypoint script (`entrypoint.sh`) handles syncing:

```bash
sync_to_workspace() {
  # Sync template files to mounted /workspace volume
  rsync -av --ignore-existing /opt/workspace-template/ /workspace/
  
  # Ensure required directories exist
  mkdir -p /workspace/{models,outputs,logs,captiv}
}
```

## Key Benefits of This Pattern

1. **RunPod Compatibility**: Follows RunPod's volume mounting conventions
2. **Persistence**: User data in `/workspace` persists across container restarts
3. **Clean Separation**: Application code at `/captiv`, user data at `/workspace`
4. **Global Access**: `captiv` command available globally in container
5. **Template System**: Fresh workspace files on first run, preserved thereafter

## Application Layout

```txt
/captiv/                    # Application installation (global)
├── src/captiv/            # Source code
├── pyproject.toml         # Project configuration
└── README.md              # Project documentation

/opt/workspace-template/   # Template files (synced to /workspace)
├── README.md              # Workspace documentation
├── models/                # AI models directory
├── outputs/               # Generated captions
├── logs/                  # Application logs
└── captiv/
    └── health.json        # Health check endpoint

/workspace/                # Mounted volume (RunPod persistent storage)
├── README.md              # Synced from template
├── models/                # User's downloaded models
├── outputs/               # User's generated captions
├── logs/                  # Runtime logs
└── captiv/
    └── health.json        # Health check endpoint
```

## Usage Notes

- The `captiv` CLI command works from any directory
- Users work in `/workspace` for data persistence
- Application updates require rebuilding the Docker image
- User data in `/workspace` persists across container restarts
- Template files are only copied if they don't already exist in `/workspace`
