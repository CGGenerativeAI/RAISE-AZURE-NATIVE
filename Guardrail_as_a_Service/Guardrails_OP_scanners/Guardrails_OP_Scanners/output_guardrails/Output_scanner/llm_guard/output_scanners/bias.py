from typing import Dict, Optional, Any

from llm_guard.transformers_helpers import pipeline_text_classification
from llm_guard.util import logger

from .base import Scanner

_model_path = (
    "valurank/distilroberta-bias",
    "laiyer/distilroberta-bias-onnx",  # ONNX model
)


class Bias(Scanner):
    """
    This class is designed to detect and evaluate potential biases in text using a pretrained model from HuggingFace.
    """

    def __init__(self, use_onnx: bool = False):
        """
        Initializes the Bias scanner with a model for bias detection.

        Parameters:
           use_onnx (bool): Whether to use ONNX instead of PyTorch for inference.
        """

        self._classifier = pipeline_text_classification(
            model=_model_path[0],
            onnx_model=_model_path[1],
            truncation=True,
            use_onnx=use_onnx,
        )

    def scan(self, prompt: str, output: str, **kwargs: Optional[Dict[str, Any]]) -> (str, bool, float):
        if output.strip() == "":
            return output, True, 0.0
        
        default_kwargs  = {"threshold":0.75}
        kwargs_1 = kwargs.get("Bias",default_kwargs)
        threshold = kwargs_1.get("threshold",0.75)

        classifier_output = self._classifier(output)
        score = round(
            classifier_output[0]["score"]
            if classifier_output[0]["label"] == "BIASED"
            else 1 - classifier_output[0]["score"],
            2,
        )
        if score > threshold:
            logger.warning(
                f"Detected biased text with score: {score}, threshold: {threshold}"
            )

            return output, False, score

        logger.debug(f"Not biased result. Max score: {score}, threshold: {threshold}")

        return output, True, score
