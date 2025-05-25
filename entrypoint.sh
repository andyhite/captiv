#!/bin/bash
# RunPod entrypoint script for Captiv application

set -e

# Environment variables
export DEBIAN_FRONTEND=noninteractive
export SHELL=/bin/bash
export PATH="/venv/bin:$PATH"

export RUNPOD_POD_ID=${RUNPOD_POD_ID:-"local"}
export RUNPOD_PUBLIC_IP=${RUNPOD_PUBLIC_IP:-"127.0.0.1"}

export CAPTIV_HOST=${CAPTIV_HOST:-"0.0.0.0"}
export CAPTIV_PORT=${CAPTIV_PORT:-7860}
export CAPTIV_SHARE=${CAPTIV_SHARE:-false}
export CAPTIV_CONFIG_PATH=${CAPTIV_CONFIG_PATH:-"/workspace/.captiv"}

show_banner() {
  echo "=============================================="
  echo "  Captiv Image Captioning Application"
  echo "  RunPod Template Version"
  echo "=============================================="
  echo "Pod ID: ${RUNPOD_POD_ID}"
  echo "Public IP: ${RUNPOD_PUBLIC_IP}"
  echo "Workspace: /workspace"
  echo "=============================================="
}

sync_to_workspace() {
  echo "Syncing workspace template to /workspace..."

  # Check if workspace is available (RunPod mounts this)
  if [ ! -d "/workspace" ]; then
    echo "Warning: /workspace not available, creating directory..."
    mkdir -p /workspace
  fi

  # Sync template files to workspace (only if they don't exist)
  if [ -d "/opt/workspace-template" ]; then
    echo "Copying workspace template files..."
    rsync -av --ignore-existing /opt/workspace-template/ /workspace/
  fi

  # Ensure workspace directories exist
  mkdir -p /workspace/{models,outputs,logs,captiv}

  # Create health check endpoint if it doesn't exist
  if [ ! -f "/workspace/captiv/health.json" ]; then
    mkdir -p /workspace/captiv
    echo '{"status": "healthy", "service": "captiv", "timestamp": "'$(date -Iseconds)'"}' >/workspace/captiv/health.json
  fi

  # Set proper permissions
  chmod -R 755 /workspace 2>/dev/null || true

  echo "Workspace sync complete."
}

check_gpu() {
  echo "Checking GPU availability..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits |
      while IFS=, read -r name total used free; do
        echo "  GPU: $name"
        echo "  Memory: ${used}MB / ${total}MB used (${free}MB free)"
      done
    echo ""
    return 0
  else
    echo "Warning: nvidia-smi not found. Running in CPU-only mode."
    echo ""
    return 1
  fi
}

start_jupyter() {
  echo "Starting Jupyter Lab..."

  if ! command -v jupyter >/dev/null 2>&1; then
    echo "Jupyter not available, skipping..."
    return 0
  fi

  # Ensure logs directory exists
  mkdir -p /workspace/logs

  cd /workspace
  nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token="" \
    --ServerApp.password="" \
    --ServerApp.allow_origin="*" \
    --ServerApp.base_url="/lab" >/workspace/logs/jupyter.log 2>&1 &

  echo "Jupyter Lab started on port 8888"
}

start_ssh() {
  echo "Starting SSH service..."

  if [ ! -w "/etc/ssh" ] || ! command -v service >/dev/null 2>&1; then
    echo "SSH not available, skipping..."
    return 0
  fi

  if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
  fi

  service ssh start
  echo "SSH service started on port 22"
}

start_nginx() {
  echo "Starting NGINX proxy..."

  if ! command -v service >/dev/null 2>&1; then
    echo "NGINX not available, skipping..."
    return 0
  fi

  service nginx start
  echo "NGINX proxy started on port 80"
}

launch_cli() {
  echo "Starting Captiv CLI mode..."
  echo "Available commands:"
  echo "  captiv --help              # Show help"
  echo "  captiv model list          # List available models"
  echo "  captiv caption generate    # Generate captions"
  echo "  captiv gui launch          # Launch GUI from CLI"
  echo ""

  # Work from workspace directory for user convenience
  cd /workspace

  if [ $# -gt 0 ]; then
    echo "Executing: captiv $*"
    exec captiv "$@"
  else
    echo "Starting interactive shell..."
    echo "Type 'captiv --help' for available commands"
    echo "Working directory: /workspace"
    exec bash
  fi
}

launch_gui() {
  local share_mode="$1"

  echo "Starting Captiv GUI..."
  check_gpu

  if [ "$share_mode" = "public" ] || [ "$CAPTIV_SHARE" = "true" ]; then
    export CAPTIV_SHARE="true"
    echo "GUI will be accessible publicly via Gradio share URL"
  else
    export CAPTIV_SHARE="false"
    echo "GUI will be accessible at http://${CAPTIV_HOST}:${CAPTIV_PORT}"
  fi

  echo "Starting GUI server..."
  echo "Host: ${CAPTIV_HOST}"
  echo "Port: ${CAPTIV_PORT}"
  echo "Share: ${CAPTIV_SHARE}"
  echo ""

  # Work from workspace directory
  cd /workspace

  # Launch the GUI (captiv is globally available)
  exec captiv gui launch
}

launch_api() {
  echo "Starting Captiv API server..."
  check_gpu

  echo "API will be accessible at http://${CAPTIV_HOST}:${CAPTIV_PORT}"
  echo "Health check: http://${CAPTIV_HOST}:${CAPTIV_PORT}/health"
  echo "Caption endpoint: http://${CAPTIV_HOST}:${CAPTIV_PORT}/api/caption"
  echo ""

  # Work from workspace directory
  cd /workspace

  # Launch the API server
  exec python -m captiv.api.server
}

pre_start() {
  echo "Running pre-start tasks..."

  # Sync workspace template to mounted volume
  sync_to_workspace

  # Start standard RunPod services unless disabled
  if [ -z "$RUNPOD_STOP_AUTO" ]; then
    start_ssh
    start_nginx
    start_jupyter
  else
    echo "Auto-start disabled by RUNPOD_STOP_AUTO"
  fi

  echo "Pre-start tasks complete."
}

show_usage() {
  echo "Captiv RunPod Entrypoint"
  echo "Usage: $0 [mode] [options]"
  echo ""
  echo "Modes:"
  echo "  cli                 Launch CLI interface (default)"
  echo "  gui                 Launch GUI interface"
  echo "  gui-public          Launch GUI with public sharing"
  echo "  api                 Launch API server for RunPod"
  echo "  jupyter             Start Jupyter Lab only"
  echo "  bash                Interactive bash shell"
  echo "  help                Show this help"
  echo ""
  echo "Environment Variables:"
  echo "  CAPTIV_HOST         GUI host (default: 0.0.0.0)"
  echo "  CAPTIV_PORT         GUI port (default: 7860)"
  echo "  CAPTIV_SHARE        Enable public sharing (true/false)"
  echo "  RUNPOD_STOP_AUTO    Disable auto-start of services"
  echo ""
  echo "Examples:"
  echo "  $0 cli model list   # List available models"
  echo "  $0 gui              # Start GUI"
  echo "  $0 gui-public       # Start GUI with sharing"
  echo "  $0 api              # Start API server"
  echo "  $0 bash             # Interactive shell"
}

main() {
  local mode="${1:-cli}"

  show_banner
  pre_start

  echo "Starting mode: $mode"
  echo ""

  case "$mode" in
  "cli")
    shift
    launch_cli "$@"
    ;;
  "gui")
    launch_gui "local"
    ;;
  "gui-public")
    launch_gui "public"
    ;;
  "api")
    launch_api
    ;;
  "jupyter")
    echo "Jupyter Lab is running on port 8888"
    echo "Access it at: http://localhost:8888/lab"
    if [ -f "/workspace/logs/jupyter.log" ]; then
      tail -f /workspace/logs/jupyter.log
    else
      echo "Jupyter log not found, keeping container running..."
      sleep infinity
    fi
    ;;
  "bash")
    echo "Starting interactive bash shell..."
    echo "Captiv is available via 'captiv' command"
    echo "Working directory: /workspace"
    cd /workspace
    exec bash
    ;;
  "help" | "--help" | "-h")
    show_usage
    ;;
  *)
    echo "Unknown mode: $mode"
    echo ""
    show_usage
    exit 1
    ;;
  esac
}

main "$@"
