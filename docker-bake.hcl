# Docker Bake configuration for Captiv - RunPod Production Optimized

group "default" {
  targets = ["captiv-runpod"]
}

group "production" {
  targets = ["captiv-runpod"]
}

variable "IMAGE_NAME" {
  default = "andyhite/captiv"
}

variable "IMAGE_TAG" {
  default = "latest"
}

variable "PYTHON_VERSION" {
  default = "3.12"
}

variable "POETRY_VERSION" {
  default = "2.1.3"
}

# Main RunPod production target
target "captiv-runpod" {
  dockerfile = "Dockerfile"
  context = "."
  tags = [
    "${IMAGE_NAME}:${IMAGE_TAG}",
    "${IMAGE_NAME}:runpod",
    "${IMAGE_NAME}:production"
  ]
  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
    POETRY_VERSION = "${POETRY_VERSION}"
  }
  platforms = ["linux/amd64"]
  cache-from = ["type=gha"]
  cache-to = ["type=gha,mode=max"]
}

# Tagged release target
target "captiv-runpod-release" {
  inherits = ["captiv-runpod"]
  tags = [
    "${IMAGE_NAME}:${IMAGE_TAG}",
    "${IMAGE_NAME}:${IMAGE_TAG}-runpod"
  ]
}
