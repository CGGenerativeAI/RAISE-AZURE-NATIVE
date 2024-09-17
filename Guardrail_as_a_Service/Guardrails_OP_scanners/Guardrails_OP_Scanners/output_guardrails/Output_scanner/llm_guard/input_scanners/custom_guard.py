import re
from typing import Dict, List, Optional, Any

from llm_guard.transformers_helpers import pipeline_text_classification
from llm_guard.util import calculate_risk_score, logger, remove_markdown

from .base import Scanner

MODEL_SM = {
    "path": "vishnun/codenlbert-sm",
    "onnx_path": "protectai/vishnun-codenlbert-sm-onnx",
    "max_length": 128,
}

MODEL_TINY = {
    "path": "vishnun/codenlbert-tiny",
    "onnx_path": "protectai/vishnun-codenlbert-tiny-onnx",
    "max_length": 128, 
}

ALL_MODELS = [MODEL_SM, MODEL_TINY]

class CustomGuard(Scanner):
    """
    A scanner that detects if input is code and blocks it.
    """

    def __init__(
        self,
        model: Optional[Dict] = None,
        use_onnx: bool = False,
    ):
        """
        Initializes the CustomGuard scanner.

        Args:
            model (Dict, optional): Model to use for zero-shot classification. Default is codenlbert-sm.
            use_onnx (bool, optional): Whether to use ONNX for inference. Default is False.

        """

        if model is None:
            model = MODEL_SM

        if model not in ALL_MODELS:
            raise ValueError(f"Model must be in the list of allowed: {ALL_MODELS}")


        self._classifier = pipeline_text_classification(
            model=model["path"],
            onnx_model=model["onnx_path"],
            use_onnx=use_onnx,
            max_length=model["max_length"],
            truncation=True,
        )

    def scan(self, prompt: str,**kwargs: Optional[Dict[str, Any]]) -> (str, bool, float):
        if prompt.strip() == "":
            return prompt, True, 0.0

        default_kwargs  = {"threshold":0.8}
        kwargs_1 = kwargs.get("CustomGuard",default_kwargs)

        threshold = kwargs_1.get("threshold",0.8)

        # Hack: Improve accuracy
        new_prompt = remove_markdown(prompt)  # Remove markdown
        new_prompt = re.sub(r"\d+\.\s+|[-*•]\s+", "", new_prompt)  # Remove list markers
        new_prompt = re.sub(r"\d+", "", new_prompt)  # Remove numbers
        new_prompt = re.sub(r'\.(?!\d)(?=[\s\'"“”‘’)\]}]|$)', "", new_prompt)  # Remove periods

        result = self._classifier(new_prompt)[0]
        score = round(
            result["score"] if result["label"] in "CODE" else 1 - result["score"],
            2,
        )

        if score > threshold:
            logger.warning(
                f"Detected code in the text, score={score}, threshold={threshold}, text={new_prompt}"
            )

            return prompt, False, calculate_risk_score(score, threshold)

        logger.debug(
            f"No code detected in the text, score={score}, threshold={threshold}, text={new_prompt}"
        )

        return prompt, True, score