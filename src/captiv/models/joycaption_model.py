"""
JoyCaption model implementation for advanced image captioning.

This module provides the JoyCaption model, which is based on the LLaVA architecture
and fine-tuned specifically for generating high-quality, detailed image captions.
JoyCaption excels at producing natural, descriptive captions and supports various
captioning modes and styles.

JoyCaption uses a chat-based interface and supports extensive prompt customization
options for controlling the style, content, and format of generated captions.

Features:
- Multiple captioning modes (descriptive, creative, technical, etc.)
- Extensive prompt customization options
- Support for different output formats (tags, descriptions, etc.)
- Chat-based prompt formatting

Reference:
    Based on LLaVA architecture with specialized fine-tuning for image captioning
"""

from typing import Any

from transformers.models.llava import LlavaForConditionalGeneration, LlavaProcessor

from .base_model import create_model


class JoyCaptionModel(
    create_model(
        model_class=LlavaForConditionalGeneration,
        processor_class=LlavaProcessor,
        default_variant="joycaption-beta-one",
        variants={
            "joycaption-alpha-two": {
                "huggingface_id": "fancyfeast/llama-joycaption-alpha-two-hf-llava",
                "description": "JoyCaption model (alpha two version) for image captioning",  # noqa: E501
                "default_mode": "default",
            },
            "joycaption-beta-one": {
                "huggingface_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
                "description": "JoyCaption model (beta one version) for image captioning",  # noqa: E501
                "default_mode": "default",
            },
        },
        modes={
            "descriptive_formal": "Generate a formal, detailed description of this image",  # noqa: E501
            "descriptive_casual": "Write a descriptive caption for this image in a casual tone.",  # noqa: E501
            "straightforward": 'Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what\'s absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.',  # noqa: E501
            "stable_diffusion": "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",  # noqa: E501
            "midjourney": "Write a MidJourney prompt for this image.",
            "danbooru": "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",  # noqa: E501
            "e621": "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",  # noqa: E501
            "rule34": "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",  # noqa: E501
            "booru": "Write a list of Booru-like tags for this image.",
            "art_critic": "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",  # noqa: E501
            "product_listing": "Write a caption for this image as though it were a product listing.",  # noqa: E501
            "social_media": "Write a caption for this image as if it were being used for a social media post.",  # noqa: E501
            "creative": "Create an imaginative, creative caption for this image.",
            "technical": "Provide a technical analysis of this image with precise details.",  # noqa: E501
            "poetic": "Write a poetic description of this image using vivid imagery.",
            "storytelling": "Create a short story inspired by this image.",
            "emotional": "Describe the emotional impact and mood of this image.",
            "humorous": "Write a humorous caption for this image.",
            "seo_friendly": "Create an SEO-friendly description for this image.",
            "accessibility": "Write an accessibility-focused description of this image.",  # noqa: E501
            "concise": "Provide a concise, brief description of this image.",
            "detailed": "Create a highly detailed description of this image.",
            "default": "Describe this image.",
        },
        prompt_options={
            "character_name": "If there is a person/character in the image you must refer to them as {character_name}.",  # noqa: E501
            "exclude_immutable": "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",  # noqa: E501
            "include_lighting": "Include information about lighting.",
            "include_camera_angle": "Include information about camera angle.",
            "include_watermark": "Include information about whether there is a watermark or not.",  # noqa: E501
            "include_jpeg_artifacts": "Include information about whether there are JPEG artifacts or not.",  # noqa: E501
            "include_camera_details": "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",  # noqa: E501
            "keep_pg": "Do NOT include anything sexual; keep it PG.",
            "exclude_resolution": "Do NOT mention the image's resolution.",
            "include_quality": "You MUST include information about the subjective aesthetic quality of the image from low to very high.",  # noqa: E501
            "include_composition": "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",  # noqa: E501
            "exclude_text": "Do NOT mention any text that is in the image.",
            "include_depth_of_field": "Specify the depth of field and whether the background is in focus or blurred.",  # noqa: E501
            "include_lighting_source": "If applicable, mention the likely use of artificial or natural lighting sources.",  # noqa: E501
            "exclude_ambiguity": "Do NOT use any ambiguous language.",
            "include_content_rating": "Include whether the image is sfw, suggestive, or nsfw.",  # noqa: E501
            "focus_important_elements": "ONLY describe the most important elements of the image.",  # noqa: E501
            "exclude_artist_info": "If it is a work of art, do not include the artist's name or the title of the work.",  # noqa: E501
            "include_orientation": "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",  # noqa: E501
            "use_vulgar_language": 'Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.',  # noqa: E501
            "use_blunt_phrasing": "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",  # noqa: E501
            "include_ages": "Include information about the ages of any people/characters when applicable.",  # noqa: E501
            "include_shot_type": "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",  # noqa: E501
            "exclude_mood": "Do not mention the mood/feeling/etc of the image.",
            "include_vantage_height": "Explicitly specify the vantage height (eye-level, low-angle worm's-eye, bird's-eye, drone, rooftop, etc.).",  # noqa: E501
            "mention_watermark": "If there is a watermark, you must mention it.",
            "avoid_meta_phrases": 'Your response will be used by a text-to-image model, so avoid useless meta phrases like "This image shows…", "You are looking at...", etc.',  # noqa: E501
        },
    )
):
    def create_chat(self, prompt: str | None) -> str:
        """
        Create a chat conversation format for the LLaVA-based model.

        Args:
            prompt: The user prompt for image captioning

        Returns:
            Formatted chat conversation string
        """
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]

        return self._processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )

    def process_inputs(self, image, prompt) -> dict[str, Any]:
        """
        Process image and prompt inputs for JoyCaption model.

        Overrides the base method to use chat formatting and ensure
        proper dtype conversion for pixel values.

        Args:
            image: PIL Image object
            prompt: Text prompt for captioning

        Returns:
            Processed inputs ready for model generation
        """
        inputs = super().process_inputs(image, self.create_chat(prompt))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        return inputs

    def generate_ids(self, inputs: Any, **kwargs) -> Any:
        """
        Generate token IDs for JoyCaption model.

        Overrides the base method to extract only the newly generated tokens,
        excluding the input prompt tokens.

        Args:
            inputs: Processed model inputs
            **kwargs: Generation parameters

        Returns:
            Generated token IDs (excluding input tokens)
        """
        generated_ids = super().generate_ids(inputs, **kwargs)

        return generated_ids[inputs["input_ids"].shape[1] :]

    def decode_caption(self, generated_ids: Any) -> str:
        """
        Decode generated token IDs into caption text for JoyCaption.

        Uses the processor's tokenizer with specific settings optimized
        for JoyCaption output formatting.

        Args:
            generated_ids: Generated token IDs from the model

        Returns:
            Decoded and cleaned caption text
        """
        return self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
