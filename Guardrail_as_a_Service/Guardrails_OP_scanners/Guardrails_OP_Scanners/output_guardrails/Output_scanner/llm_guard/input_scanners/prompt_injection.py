from typing import Dict, List, Optional, Any

from llm_guard.transformers_helpers import pipeline_text_classification
from llm_guard.util import calculate_risk_score, logger

from .base import Scanner

# This model is proprietary but open source.
MODEL_LAIYER = {
    "path": "laiyer/deberta-v3-base-prompt-injection",
    "onnx_path": "laiyer/deberta-v3-base-prompt-injection",  # extract from onnx folder
    "label": "INJECTION",
    "max_length": 512,
}

ALL_MODELS = [
    MODEL_LAIYER,
]


class PromptInjection(Scanner):
    """
    A prompt injection scanner based on HuggingFace model. It is used to
    detect if a prompt is attempting to perform an injection attack.
    """

    def __init__(
        self, models: Optional[List[Dict]] = None, use_onnx: bool = False
    ):
        """
        Initializes PromptInjection with a model.

        Parameters:
            models (List[Dict], optional): Chosen models to classify prompt. Default is Laiyer's one..
            use_onnx (bool): Whether to use ONNX for inference. Defaults to False.

        Raises:
            ValueError: If non-existent models were provided.
        """


        if models is None:
            models = ALL_MODELS

        for model in models:
            if model not in ALL_MODELS:
                raise ValueError(f"Models must be in the list of allowed: {ALL_MODELS}")

        logger.debug(f"Prompt injection models: {models}")

        pipelines = {}
        for model in models:
            try:
                pipelines[model["path"]] = pipeline_text_classification(
                    model=model["path"],
                    use_onnx=use_onnx,
                    onnx_model=model["onnx_path"],
                    truncation=True,
                    max_length=model["max_length"],
                )
            except Exception as err:
                logger.error(f"Failed to load model: {err}")

        self._text_classification_pipelines = pipelines

    def scan(self, prompt: str, **kwargs: Optional[Dict[str, Any]]) -> (str, bool, float):

        if prompt.strip() == "":
            return prompt, True, 0.0
        
        default_kwargs  = {"threshold":0.9}
        kwargs_1 = kwargs.get("PromptInjection",default_kwargs)

        threshold = kwargs_1.get("threshold",0.9)
        
        # threshold = kwargs.get("Input_PromptInjection_threshold",0.9)

        for model_path in self._text_classification_pipelines:
            try:
                result = self._text_classification_pipelines[model_path](prompt)
            except Exception as err:
                logger.error(f"Failed to get output of the model: {err}")
                return prompt, True, 0.0

            model_label = ALL_MODELS[[m["path"] for m in ALL_MODELS].index(model_path)]["label"]

            injection_score = round(
                result[0]["score"] if result[0]["label"] == model_label else 1 - result[0]["score"],
                2,
            )

            if injection_score > threshold:
                logger.warning(
                    f"Detected prompt injection using {model_path} with score: {injection_score}"
                )

                return prompt, False, round(calculate_risk_score(injection_score, threshold),2)

            logger.debug(
                f"No prompt injection detected using {model_path}, score: {injection_score}"
            )

        return prompt, True, round(injection_score,2)
