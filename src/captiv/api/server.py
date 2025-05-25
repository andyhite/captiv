"""
FastAPI server for RunPod image captioning service.

This module provides a REST API endpoint for generating image captions using JoyCaption
models in RunPod environments.
"""

import io
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from captiv.models.joycaption_model import JoyCaptionModel
from captiv.services.exceptions import CaptivServiceError

app = FastAPI(
    title="Captiv JoyCaption API",
    description="Image captioning API using JoyCaption models",
    version="1.0.0",
)

# Global model instance (loaded on startup)
model_instance: JoyCaptionModel | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_instance
    try:
        logger.info("Loading JoyCaption model...")
        model_instance = JoyCaptionModel("joycaption-beta-one")
        logger.info("JoyCaption model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "healthy", "model": "joycaption-beta-one"}


@app.post("/api/caption")
async def generate_caption(
    image: UploadFile = File(...),
    model_variant: str = Form("joycaption-beta-one"),
    mode: str = Form("default"),
    prompt: str = Form(""),
    generation_params: str = Form("{}"),
):
    """
    Generate a caption for an uploaded image.

    Args:
        image: Image file to caption
        model_variant: JoyCaption model variant to use
        mode: Captioning mode
        prompt: Custom prompt (overrides mode if provided)
        generation_params: JSON string of generation parameters

    Returns:
        JSON response with generated caption
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse generation parameters
        try:
            gen_params = json.loads(generation_params) if generation_params else {}
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail="Invalid generation_params JSON"
            ) from e

        # Determine prompt or mode
        prompt_or_mode = prompt if prompt else mode

        # Generate caption
        logger.info(f"Generating caption with mode/prompt: {prompt_or_mode}")
        caption = model_instance.caption_image(
            image_input=pil_image, prompt_or_mode=prompt_or_mode, **gen_params
        )

        logger.info(f"Caption generated: {caption[:100]}...")

        return JSONResponse(
            {
                "caption": caption,
                "model_variant": model_variant,
                "mode": mode,
                "prompt": prompt,
            }
        )

    except CaptivServiceError as e:
        logger.error(f"Captiv service error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/api/models")
async def list_models():
    """List available models and variants."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "models": {
            "joycaption": {
                "variants": list(JoyCaptionModel.get_variants().keys()),
                "modes": list(JoyCaptionModel.get_modes().keys()),
                "prompt_options": list(JoyCaptionModel.get_prompt_options().keys()),
            }
        }
    }


@app.get("/api/modes/{model_name}")
async def get_model_modes(model_name: str):
    """Get available modes for a specific model."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if model_name.lower() != "joycaption":
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model": model_name,
        "modes": JoyCaptionModel.get_modes(),
        "prompt_options": JoyCaptionModel.get_prompt_options(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
