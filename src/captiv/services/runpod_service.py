"""
RunPod service for managing remote GPU instances for model inference.

This service handles the creation, management, and communication with RunPod instances
for running image captioning models remotely. It provides a seamless interface for
offloading compute-intensive model operations to cloud GPUs.
"""

import json
import time
from typing import Any

import requests
from loguru import logger

from captiv.services.exceptions import RunPodError


class RunPodService:
    """Service for managing RunPod instances and remote model inference."""

    def __init__(self, api_key: str, template_id: str | None = None):
        """
        Initialize the RunPod service.

        Args:
            api_key: RunPod API key for authentication
            template_id: Optional template ID for pod creation
        """
        self.api_key = api_key
        self.template_id = template_id
        self.base_url = "https://api.runpod.ai/graphql"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.active_pod_id: str | None = None
        self.pod_endpoint: str | None = None

    def _make_request(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make a GraphQL request to the RunPod API.

        Args:
            query: GraphQL query string
            variables: Optional variables for the query

        Returns:
            Response data from the API

        Raises:
            RunPodError: If the API request fails
        """
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            if "errors" in data:
                error_msg = "; ".join([error["message"] for error in data["errors"]])
                raise RunPodError(f"RunPod API error: {error_msg}")

            return data.get("data", {})

        except requests.RequestException as e:
            raise RunPodError(f"Failed to communicate with RunPod API: {str(e)}") from e

    def create_pod(
        self,
        name: str = "captiv-joycaption",
        gpu_type: str = "NVIDIA RTX A4000",
        container_disk_size: int = 50,
        volume_disk_size: int = 50,
        ports: str = "7860/http,8888/http,22/tcp",
    ) -> str:
        """
        Create a new RunPod instance.

        Args:
            name: Name for the pod
            gpu_type: GPU type to use
            container_disk_size: Container disk size in GB
            volume_disk_size: Volume disk size in GB
            ports: Port configuration string

        Returns:
            Pod ID of the created instance

        Raises:
            RunPodError: If pod creation fails
        """
        logger.info(f"Creating RunPod instance: {name}")

        # Use template if provided, otherwise create from scratch
        if self.template_id:
            query = """
            mutation createPodFromTemplate($input: PodRentInterruptableInput!) {
                podRentInterruptable(input: $input) {
                    id
                    desiredStatus
                    imageName
                    env
                    machineId
                    machine {
                        podHostId
                    }
                }
            }
            """
            variables = {
                "input": {
                    "name": name,
                    "templateId": self.template_id,
                    "gpuTypeId": gpu_type,
                    "containerDiskInGb": container_disk_size,
                    "volumeInGb": volume_disk_size,
                    "ports": ports,
                }
            }
        else:
            # Create pod with Captiv Docker image
            query = """
            mutation createPod($input: PodRentInterruptableInput!) {
                podRentInterruptable(input: $input) {
                    id
                    desiredStatus
                    imageName
                    env
                    machineId
                    machine {
                        podHostId
                    }
                }
            }
            """
            variables = {
                "input": {
                    "name": name,
                    "imageName": "captiv/joycaption:latest",  # Will be built and pushed
                    "gpuTypeId": gpu_type,
                    "containerDiskInGb": container_disk_size,
                    "volumeInGb": volume_disk_size,
                    "ports": ports,
                    "env": [
                        {"key": "JUPYTER_PASSWORD", "value": "captiv123"},
                        {"key": "RUNPOD_POD_ID", "value": "{{POD_ID}}"},
                    ],
                }
            }

        try:
            data = self._make_request(query, variables)
            pod_data = data.get("podRentInterruptable")

            if not pod_data or not pod_data.get("id"):
                raise RunPodError("Failed to create pod: No pod ID returned")

            pod_id = pod_data["id"]
            self.active_pod_id = pod_id

            logger.info(f"Pod created successfully: {pod_id}")
            return pod_id

        except Exception as e:
            raise RunPodError(f"Failed to create RunPod instance: {str(e)}") from e

    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> dict[str, Any]:
        """
        Wait for a pod to be ready and return its details.

        Args:
            pod_id: ID of the pod to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Pod details including endpoint information

        Raises:
            RunPodError: If pod doesn't become ready within timeout
        """
        logger.info(f"Waiting for pod {pod_id} to be ready...")

        query = """
        query getPod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                desiredStatus
                lastStatusChange
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
                machine {
                    podHostId
                }
            }
        }
        """

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data = self._make_request(query, {"podId": pod_id})
                pod_data = data.get("pod")

                if not pod_data:
                    raise RunPodError(f"Pod {pod_id} not found")

                status = pod_data.get("desiredStatus")
                runtime = pod_data.get("runtime")

                if status == "RUNNING" and runtime and runtime.get("ports"):
                    # Find the HTTP port for our service
                    ports = runtime.get("ports", [])
                    http_port = None

                    for port in ports:
                        if (
                            port.get("privatePort") == 7860
                            and port.get("type") == "http"
                        ):
                            if port.get("isIpPublic"):
                                http_port = f"https://{pod_id}-7860.proxy.runpod.net"
                            else:
                                http_port = (
                                    f"http://{port.get('ip')}:{port.get('publicPort')}"
                                )
                            break

                    if http_port:
                        self.pod_endpoint = http_port
                        logger.info(f"Pod {pod_id} is ready at {http_port}")
                        return pod_data

                logger.debug(f"Pod {pod_id} status: {status}, waiting...")
                time.sleep(10)

            except Exception as e:
                logger.warning(f"Error checking pod status: {str(e)}")
                time.sleep(10)

        raise RunPodError(f"Pod {pod_id} did not become ready within {timeout} seconds")

    def stop_pod(self, pod_id: str) -> bool:
        """
        Stop a running pod.

        Args:
            pod_id: ID of the pod to stop

        Returns:
            True if successful

        Raises:
            RunPodError: If stopping the pod fails
        """
        logger.info(f"Stopping pod {pod_id}")

        query = """
        mutation stopPod($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """

        variables = {"input": {"podId": pod_id}}

        try:
            data = self._make_request(query, variables)
            result = data.get("podStop")

            if result and result.get("desiredStatus") == "EXITED":
                logger.info(f"Pod {pod_id} stopped successfully")
                if self.active_pod_id == pod_id:
                    self.active_pod_id = None
                    self.pod_endpoint = None
                return True
            else:
                raise RunPodError(f"Failed to stop pod {pod_id}")

        except Exception as e:
            raise RunPodError(f"Failed to stop pod {pod_id}: {str(e)}") from e

    def terminate_pod(self, pod_id: str) -> bool:
        """
        Terminate a pod (permanent deletion).

        Args:
            pod_id: ID of the pod to terminate

        Returns:
            True if successful

        Raises:
            RunPodError: If terminating the pod fails
        """
        logger.info(f"Terminating pod {pod_id}")

        query = """
        mutation terminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input) {
                id
            }
        }
        """

        variables = {"input": {"podId": pod_id}}

        try:
            data = self._make_request(query, variables)
            result = data.get("podTerminate")

            if result and result.get("id"):
                logger.info(f"Pod {pod_id} terminated successfully")
                if self.active_pod_id == pod_id:
                    self.active_pod_id = None
                    self.pod_endpoint = None
                return True
            else:
                raise RunPodError(f"Failed to terminate pod {pod_id}")

        except Exception as e:
            raise RunPodError(f"Failed to terminate pod {pod_id}: {str(e)}") from e

    def generate_caption_remote(
        self,
        image_data: bytes,
        model_variant: str = "joycaption-beta-one",
        mode: str = "default",
        prompt: str | None = None,
        **generation_params,
    ) -> str:
        """
        Generate a caption using the remote RunPod instance.

        Args:
            image_data: Image data as bytes
            model_variant: JoyCaption model variant to use
            mode: Captioning mode
            prompt: Custom prompt (overrides mode)
            **generation_params: Additional generation parameters

        Returns:
            Generated caption text

        Raises:
            RunPodError: If remote caption generation fails
        """
        if not self.pod_endpoint:
            raise RunPodError("No active pod endpoint available")

        logger.info(f"Generating caption remotely using {model_variant}")

        # Prepare the request payload
        files = {"image": ("image.jpg", image_data, "image/jpeg")}
        data = {
            "model_variant": model_variant,
            "mode": mode,
            "prompt": prompt or "",
            "generation_params": json.dumps(generation_params),
        }

        try:
            # Make request to the pod's caption endpoint
            response = requests.post(
                f"{self.pod_endpoint}/api/caption",
                files=files,
                data=data,
                timeout=120,  # Allow time for model inference
            )
            response.raise_for_status()

            result = response.json()

            if "error" in result:
                raise RunPodError(
                    f"Remote caption generation failed: {result['error']}"
                )

            caption = result.get("caption")
            if not caption:
                raise RunPodError("No caption returned from remote service")

            logger.info("Caption generated successfully")
            return caption

        except requests.RequestException as e:
            raise RunPodError(f"Failed to communicate with remote pod: {str(e)}") from e
        except Exception as e:
            raise RunPodError(f"Remote caption generation error: {str(e)}") from e

    def health_check(self) -> bool:
        """
        Check if the active pod is healthy and responsive.

        Returns:
            True if pod is healthy, False otherwise
        """
        if not self.pod_endpoint:
            return False

        try:
            response = requests.get(f"{self.pod_endpoint}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_pod_status(self, pod_id: str) -> dict[str, Any]:
        """
        Get the current status of a pod.

        Args:
            pod_id: ID of the pod to check

        Returns:
            Pod status information
        """
        query = """
        query getPod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                desiredStatus
                lastStatusChange
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
                machine {
                    podHostId
                }
            }
        }
        """

        data = self._make_request(query, {"podId": pod_id})
        return data.get("pod", {})
