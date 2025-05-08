"""
Model type configuration section for the Captiv GUI.
"""

import traceback  # Add this import
from typing import Any, Dict, List, Tuple

import gradio as gr
from loguru import logger  # Add this import

from captiv.services.model_manager import ModelManager, ModelType


class ModelSection:
    """Model type configuration section for selecting and configuring captioning models."""

    def __init__(self, model_manager: ModelManager):
        """Initialize the model section.

        Args:
            model_manager: The model manager instance
        """
        self.model_manager = model_manager
        self.current_model_type = self.model_manager.get_default_model()
        logger.info(
            f"ModelSection initialized. Default model type: {self.current_model_type.value}"
        )

        # UI components
        self.model_type_dropdown = None
        self.model_dropdown = None
        self.mode_dropdown = None
        self.prompt_textbox = None

        # Advanced options
        self.max_length_slider = None
        self.min_length_slider = None
        self.num_beams_slider = None
        self.temperature_slider = None
        self.top_k_slider = None
        self.top_p_slider = None
        self.repetition_penalty_slider = None

    def create_section(
        self,
    ) -> Tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Textbox, Dict[str, Any]]:
        """Create the model type configuration section UI components.

        Returns:
            Tuple containing the model type dropdown, model dropdown, mode dropdown, prompt textbox,
            and a dictionary of advanced options
        """
        logger.debug("Creating model section UI components.")
        model_type_choices = [model.value for model in ModelType]
        logger.debug(f"Model type choices: {model_type_choices}")
        self.model_type_dropdown = gr.Dropdown(
            label="Model Type",
            value=self.current_model_type.value,
            interactive=True,
            choices=model_type_choices,
        )

        model_choices = self.get_models_for_model_type(self.current_model_type)
        # Get the default variant for the current model type
        model_class = self.model_manager.get_model_class(self.current_model_type)
        default_variant = model_class.get_default_variant()
        default_model_value = (
            default_variant
            if default_variant in model_choices
            else (model_choices[0] if model_choices else None)
        )
        logger.debug(
            f"Initial model choices for {self.current_model_type.value}: {model_choices}, Default: {default_model_value}"
        )
        self.model_dropdown = gr.Dropdown(
            label="Model",
            interactive=True,
            choices=model_choices,
            value=default_model_value,
            scale=1,
        )

        mode_choices = self.get_modes_for_model(self.current_model_type)
        default_mode_value = mode_choices[0] if mode_choices else None
        logger.debug(
            f"Initial mode choices for {self.current_model_type.value}: {mode_choices}, Default: {default_mode_value}"
        )
        self.mode_dropdown = gr.Dropdown(
            label="Mode",
            interactive=True,
            choices=mode_choices,
            value=default_mode_value,
            scale=1,
        )

        # Custom prompt - initially hidden, will be shown conditionally
        self.prompt_textbox = gr.Textbox(
            label="Custom Prompt",
            placeholder="Enter a custom prompt",
            interactive=True,
            lines=2,
            visible=(default_mode_value == "custom"),
        )

        # Add event handler to show/hide prompt based on mode
        self.mode_dropdown.change(
            fn=self.on_mode_change,
            inputs=[self.mode_dropdown],
            outputs=[self.prompt_textbox],
        )

        # Options section - all flat, no accordions
        gr.Markdown("### Options")
        advanced_options = {}

        # Generation parameters
        self.max_length_slider = gr.Slider(
            label="Max Length",
            minimum=10,
            maximum=500,
            value=100,
            step=10,
            interactive=True,
        )
        advanced_options["max_length"] = self.max_length_slider

        self.min_length_slider = gr.Slider(
            label="Min Length",
            minimum=5,
            maximum=100,
            value=10,
            step=5,
            interactive=True,
        )
        advanced_options["min_length"] = self.min_length_slider

        self.num_beams_slider = gr.Slider(
            label="Num Beams",
            minimum=1,
            maximum=10,
            value=3,
            step=1,
            interactive=True,
        )
        advanced_options["num_beams"] = self.num_beams_slider

        self.temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=2.0,
            value=1.0,
            step=0.1,
            interactive=True,
        )
        advanced_options["temperature"] = self.temperature_slider

        self.top_k_slider = gr.Slider(
            label="Top K",
            minimum=1,
            maximum=100,
            value=50,
            step=1,
            interactive=True,
        )
        advanced_options["top_k"] = self.top_k_slider

        self.top_p_slider = gr.Slider(
            label="Top P",
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.1,
            interactive=True,
        )
        advanced_options["top_p"] = self.top_p_slider

        self.repetition_penalty_slider = gr.Slider(
            label="Repetition Penalty",
            minimum=1.0,
            maximum=5.0,
            value=1.0,
            step=0.1,
            interactive=True,
        )
        advanced_options["repetition_penalty"] = self.repetition_penalty_slider

        # Removed word count and length options

        return (
            self.model_type_dropdown,
            self.model_dropdown,
            self.mode_dropdown,
            self.prompt_textbox,
            advanced_options,
        )
        logger.debug("Model section UI components created.")

    def on_model_change(self, model_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Handle model type change event.

        Args:
            model_str: The selected model type string

        Returns:
            Tuple of (model dropdown update, mode dropdown update)
        """
        logger.debug(
            f"Model type change event triggered. Selected model string: '{model_str}'"
        )
        try:
            # Convert model string to ModelType
            model_type = ModelType(model_str)
            logger.info(f"Current model type changed to: {model_type.value}")
            self.current_model_type = model_type

            # Get models and modes for the selected model type
            models = self.get_models_for_model_type(model_type)
            logger.debug(f"Models for {model_type.value}: {models}")
            modes = self.get_modes_for_model(model_type)
            logger.debug(f"Modes for {model_type.value}: {modes}")

            # Set default values for model and mode dropdowns
            # Get the default variant for the current model type
            model_class = self.model_manager.get_model_class(model_type)
            default_variant = model_class.get_default_variant()
            model_value = (
                default_variant
                if default_variant in models
                else (models[0] if models else None)
            )
            mode_value = modes[0] if modes else None
            logger.debug(f"Default model: {model_value}, Default mode: {mode_value}")

            # Return updates for the dropdowns
            return (
                gr.update(choices=models, value=model_value),
                gr.update(choices=modes, value=mode_value),
            )
        except Exception as e:
            logger.error(f"Error changing model to '{model_str}': {e}")
            logger.debug(traceback.format_exc())
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
            )

    def on_mode_change(self, mode: str) -> Dict[str, Any]:
        """Handle mode change event.

        Args:
            mode: The selected mode

        Returns:
            Update for the prompt textbox visibility
        """
        logger.debug(f"Mode change event triggered. Selected mode: '{mode}'")
        is_custom_mode = mode == "custom"
        logger.info(
            f"Prompt textbox visibility set to: {is_custom_mode} (Mode: '{mode}')"
        )
        # Show prompt textbox only when mode is "custom"
        return gr.update(visible=is_custom_mode)

    def get_models_for_model_type(self, model_type: ModelType) -> List[str]:
        """Get models for the given model type.

        Args:
            model_type: The model type

        Returns:
            List of model names
        """
        logger.debug(f"Getting models for model type: {model_type.value}")
        models = self.model_manager.get_variants_for_model(model_type) or []
        logger.debug(f"Found models for {model_type.value}: {models}")
        return models

    def get_modes_for_model(self, model_type: ModelType) -> List[str]:
        """Get modes for the given model type.

        Args:
            model_type: The model type

        Returns:
            List of mode names
        """
        logger.debug(f"Getting modes for model type: {model_type.value}")
        modes = self.model_manager.get_modes_for_model(model_type) or []
        logger.debug(f"Found modes for {model_type.value}: {modes}")
        return modes

    def get_current_model_type(self) -> ModelType:
        """Get the current model type.

        Returns:
            The current model type
        """
        logger.debug(f"Getting current model type: {self.current_model_type.value}")
        return self.current_model_type
