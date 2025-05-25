"""Model configuration section for the Captiv GUI."""

import traceback
from typing import Any

import gradio as gr
from loguru import logger

from captiv.services.model_manager import ModelManager, ModelType


class ModelSection:
    """Model configuration section for selecting and configuring captioning models and
    their variants."""

    def __init__(self, model_manager: ModelManager):
        """
        Initialize the model section.

        Args:
            model_manager: The model manager instance
        """
        self.model_manager = model_manager
        self.current_model = self.model_manager.get_default_model()
        logger.info(
            f"ModelSection initialized. Default model: {self.current_model.value}"
        )

        self.model_dropdown = None
        self.model_variant_dropdown = None
        self.mode_dropdown = None
        self.prompt_textbox = None

        self.max_new_tokens_slider = None
        self.min_new_tokens_slider = None
        self.num_beams_slider = None
        self.temperature_slider = None
        self.top_k_slider = None
        self.top_p_slider = None
        self.repetition_penalty_slider = None

        self.prompt_options_checkbox_group = None
        self.character_name_textbox = None

    def create_section(
        self,
    ) -> tuple[
        gr.Dropdown,
        gr.Dropdown,
        gr.Dropdown,
        gr.Textbox,
        gr.Slider,
        gr.Slider,
        gr.Slider,
        gr.Slider,
        gr.Slider,
        gr.Slider,
        gr.Slider,
        gr.Accordion,
        gr.CheckboxGroup,
        gr.Textbox,
    ]:
        """
        Create the model configuration section UI components.

        Returns:
            Tuple containing the model dropdown, model variant dropdown, mode dropdown,
            prompt textbox, all the advanced option sliders, and a dictionary of prompt
            option checkboxes
        """
        logger.debug("Creating model section UI components.")
        model_choices = [model.value for model in ModelType]
        logger.debug(f"Model choices: {model_choices}")
        self.model_dropdown = gr.Dropdown(
            label="Model",
            value=self.current_model.value,
            interactive=True,
            choices=model_choices,
        )

        variant_choices = self.get_variants_for_model(self.current_model)
        model_class = self.model_manager.get_model_class(self.current_model)
        default_variant = model_class.DEFAULT_VARIANT
        default_variant_value = (
            default_variant
            if default_variant in variant_choices
            else (variant_choices[0] if variant_choices else None)
        )
        logger.debug(
            f"Initial model variant choices for {self.current_model.value}: {variant_choices}, Default: {default_variant_value}"  # noqa: E501
        )

        has_multiple_variants = len(variant_choices) > 1
        logger.debug(
            f"Initial model {self.current_model.value} has multiple variants: {has_multiple_variants}"  # noqa: E501
        )

        self.model_variant_dropdown = gr.Dropdown(
            label="Model Variant",
            interactive=True,
            choices=variant_choices,
            value=default_variant_value,
            scale=1,
            visible=has_multiple_variants,
        )

        mode_choices = self.get_modes_for_model(self.current_model)
        default_mode_value = mode_choices[0] if mode_choices else None
        logger.debug(
            f"Initial mode choices for {self.current_model.value} model: {mode_choices}, Default: {default_mode_value}"  # noqa: E501
        )
        self.mode_dropdown = gr.Dropdown(
            label="Mode",
            interactive=True,
            choices=mode_choices,
            value=default_mode_value,
            scale=1,
        )

        self.prompt_textbox = gr.Textbox(
            label="Custom Prompt",
            placeholder="Enter a custom prompt",
            interactive=True,
            lines=2,
            visible=(default_mode_value == "custom"),
        )

        self.mode_dropdown.change(
            fn=self.on_mode_change,
            inputs=[self.mode_dropdown],
            outputs=[self.prompt_textbox],
        )

        with gr.Accordion(
            "Prompt Options", open=False, visible=False, elem_classes="accordion"
        ) as self.prompt_options_accordion:
            options = self.get_prompt_options_for_model(self.current_model)
            logger.debug(f"Prompt options for {self.current_model.value}: {options}")

            checkbox_options = []
            for option_name, option_desc in options.items():
                if option_name != "character_name":
                    checkbox_options.append(
                        (
                            option_name.replace("_", " ").title(),
                            option_name,
                            option_desc,
                        )
                    )

            self.prompt_options_checkbox_group = gr.CheckboxGroup(
                label="Prompt Options",
                elem_classes="field",
                choices=[opt[0] for opt in checkbox_options],
                value=[],
                info="Select options to include in the prompt",
                container=False,
            )

            self.prompt_options_mapping = {
                opt[0]: (opt[1], opt[2]) for opt in checkbox_options
            }

            if "character_name" in options:
                self.character_name_textbox = gr.Textbox(
                    label="Character Name",
                    elem_classes="field",
                    placeholder="Enter character name",
                    interactive=True,
                    visible=True,
                    info=options["character_name"].replace("{name}", "[name]"),
                    container=False,
                )
            else:
                self.character_name_textbox = gr.Textbox(
                    label="Character Name",
                    elem_classes="field",
                    visible=False,
                )

        with gr.Accordion("Advanced Options", open=False, elem_classes="accordion"):
            self.max_new_tokens_slider = gr.Slider(
                elem_classes="field",
                label="Max Caption Length",
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                interactive=True,
            )

            self.min_new_tokens_slider = gr.Slider(
                elem_classes="field",
                label="Min Caption Length",
                minimum=5,
                maximum=100,
                value=10,
                step=5,
                interactive=True,
            )

            self.num_beams_slider = gr.Slider(
                elem_classes="field",
                label="Num Beams",
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                interactive=True,
            )

            self.temperature_slider = gr.Slider(
                elem_classes="field",
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
            )

            self.top_k_slider = gr.Slider(
                elem_classes="field",
                label="Top K",
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                interactive=True,
            )

            self.top_p_slider = gr.Slider(
                elem_classes="field",
                label="Top P",
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
            )

            self.repetition_penalty_slider = gr.Slider(
                elem_classes="field",
                label="Repetition Penalty",
                minimum=1.0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                interactive=True,
            )

        logger.debug("Model section UI components created.")

        return (
            self.model_dropdown,
            self.model_variant_dropdown,
            self.mode_dropdown,
            self.prompt_textbox,
            self.max_new_tokens_slider,
            self.min_new_tokens_slider,
            self.num_beams_slider,
            self.temperature_slider,
            self.top_k_slider,
            self.top_p_slider,
            self.repetition_penalty_slider,
            self.prompt_options_accordion,
            self.prompt_options_checkbox_group,
            self.character_name_textbox,
        )

    def on_model_change(
        self, model_str: str
    ) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
    ]:
        """
        Handle model change event.

        Args:
            model_str: The selected model string

        Returns:
            Tuple of (model variant dropdown update, mode dropdown update, checkbox
            group update, character name textbox update)
        """
        logger.debug(
            f"Model change event triggered. Selected model string: '{model_str}'"
        )
        try:
            model = ModelType(model_str)
            logger.info(f"Current model changed to: {model.value}")
            self.current_model = model

            model_variants = self.get_variants_for_model(model)
            logger.debug(f"Model variants for {model.value}: {model_variants}")
            modes = self.get_modes_for_model(model)
            logger.debug(f"Modes for {model.value}: {modes}")
            prompt_options = self.get_prompt_options_for_model(model)
            logger.debug(f"Prompt options for {model.value}: {prompt_options}")

            model_class = self.model_manager.get_model_class(model)
            default_variant = model_class.DEFAULT_VARIANT
            variant_value = (
                default_variant
                if default_variant in model_variants
                else (model_variants[0] if model_variants else None)
            )
            mode_value = modes[0] if modes else None
            logger.debug(
                f"Default model variant: {variant_value}, Default mode: {mode_value}"
            )

            checkbox_update, character_name_update = (
                self.update_prompt_options_for_model(model)
            )

            has_prompt_options = len(prompt_options) > 0

            logger.info(f"Changed model to: {model.value}")

            has_multiple_variants = len(model_variants) > 1
            logger.debug(
                f"Model {model.value} has multiple variants: {has_multiple_variants}"
            )

            return (
                gr.update(
                    choices=model_variants,
                    value=variant_value,
                    visible=has_multiple_variants,
                ),
                gr.update(choices=modes, value=mode_value),
                gr.update(visible=has_prompt_options),
                checkbox_update,
                character_name_update,
            )
        except Exception as e:
            logger.error(f"Error changing model to '{model_str}': {e}")
            logger.debug(traceback.format_exc())
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(visible=False),
                gr.update(choices=[], value=[]),
                gr.update(visible=False),
            )

    def on_mode_change(self, mode: str) -> dict[str, Any]:
        """
        Handle mode change event.

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
        return gr.update(visible=is_custom_mode)

    def get_variants_for_model(self, model: ModelType) -> list[str]:
        """
        Get model variants for the given model.

        Args:
            model: The model

        Returns:
            List of model variant names
        """
        logger.debug(f"Getting model variants for model: {model.value}")
        model_variants = self.model_manager.get_variants_for_model(model) or []
        logger.debug(f"Found model variants for {model.value}: {model_variants}")
        return model_variants

    def get_modes_for_model(self, model: ModelType) -> list[str]:
        """
        Get modes for the given model.

        Args:
            model: The model

        Returns:
            List of mode names
        """
        logger.debug(f"Getting modes for model: {model.value}")
        modes = self.model_manager.get_modes_for_model(model) or []
        logger.debug(f"Found modes for {model.value}: {modes}")
        return modes

    def get_prompt_options_for_model(self, model: ModelType) -> dict[str, str]:
        """
        Get prompt options for the given model.

        Args:
            model: The model

        Returns:
            Dictionary of prompt option names and descriptions
        """
        logger.debug(f"Getting prompt options for model: {model.value}")
        options = self.model_manager.get_prompt_option_details(model) or {}
        logger.debug(f"Found prompt options for {model.value}: {options}")
        return options

    def update_prompt_options_for_model(
        self, model: ModelType
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Update prompt options for the given model.

        Args:
            model: The model

        Returns:
            Tuple of (checkbox group update, character name textbox update)
        """
        logger.debug(f"Updating prompt options for model: {model.value}")

        options = self.get_prompt_options_for_model(model)
        logger.debug(f"Prompt options for {model.value}: {options}")

        checkbox_options = []
        for option_name, option_desc in options.items():
            if option_name != "character_name":
                checkbox_options.append(
                    (option_name.replace("_", " ").title(), option_name, option_desc)
                )

        checkbox_options.sort(key=lambda x: x[0])

        if model == ModelType.JOYCAPTION and not checkbox_options:
            logger.warning(
                f"No prompt options found for {model.value}, adding examples"
            )

            example_options = [
                (
                    "Include Camera Angle",
                    "include_camera_angle",
                    "Include information about camera angle.",
                ),
                (
                    "Include Lighting",
                    "include_lighting",
                    "Include information about lighting.",
                ),
                (
                    "Include Watermark",
                    "include_watermark",
                    "Include information about whether there is a watermark or not.",
                ),
                (
                    "Include JPEG Artifacts",
                    "include_jpeg_artifacts",
                    "Include information about whether there are JPEG artifacts or not.",  # noqa: E501
                ),
            ]

            example_options.sort(key=lambda x: x[0])
            checkbox_options.extend(example_options)

        self.prompt_options_mapping = {
            opt[0]: (opt[1], opt[2]) for opt in checkbox_options
        }

        checkbox_update = gr.update(
            choices=[opt[0] for opt in checkbox_options],
            value=[],
            visible=bool(checkbox_options),
        )

        if "character_name" in options:
            character_name_update = gr.update(
                visible=True,
                info=options["character_name"].replace("{name}", "[name]"),
            )
        elif model == ModelType.JOYCAPTION:
            character_name_update = gr.update(
                visible=True,
                info="If there is a person/character in the image, refer to them as [name].",  # noqa: E501
            )
        else:
            character_name_update = gr.update(visible=False)

        return checkbox_update, character_name_update

    def get_current_model(self) -> ModelType:
        """
        Get the current model.

        Returns:
            The current model
        """
        logger.debug(f"Getting current model: {self.current_model.value}")
        return self.current_model
