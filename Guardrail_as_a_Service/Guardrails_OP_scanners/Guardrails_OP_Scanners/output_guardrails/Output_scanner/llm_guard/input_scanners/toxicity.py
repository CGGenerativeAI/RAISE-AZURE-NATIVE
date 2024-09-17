from typing import Dict, List, Optional, Any

from llm_guard.transformers_helpers import pipeline_text_classification
from llm_guard.util import calculate_risk_score, logger

from .base import Scanner

_model_path = (
    "unitary/unbiased-toxic-roberta",
    "laiyer/unbiased-toxic-roberta-onnx",  # ONNX model
)
_toxic_labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]


class Toxicity(Scanner):
    """
    A toxicity scanner that uses a pretrained Hugging Face model to assess the toxicity of a given text.

    If the toxicity score is less than a predefined threshold, the text is considered non-toxic. Otherwise, it is
    considered toxic.
    """

    def __init__(self,  use_onnx: bool = False):
        """
        Initializes Toxicity class.

        Parameters:
           use_onnx (bool): Whether to use ONNX for inference. Default is False.

        Raises:
           None.
        """

        self._pipeline = pipeline_text_classification(
            model=_model_path[0],
            onnx_model=_model_path[1],
            top_k=None,
            use_onnx=use_onnx,
            truncation=True,
        )

    def scan(self, prompt: str, **kwargs: Optional[Dict[str, Any]]) -> (str, bool, float):
        if prompt.strip() == "":
            return prompt, True, 0.0
        
        default_kwargs  = {"threshold":0.6}
        kwargs_1 = kwargs.get("Toxicity",default_kwargs)

        threshold = kwargs_1.get("threshold",0.6)
        
        # threshold = kwargs.get("Input_Toxicity_threshold",0.6)

        results = self._pipeline(prompt)

        highest_toxicity_score = 0.0
        toxicity_above_threshold = []
        for result in results[0]:
            if result["label"] not in _toxic_labels:
                continue

            if result["score"] > threshold:
                toxicity_above_threshold.append(result)
            if result["score"] > highest_toxicity_score:
                highest_toxicity_score = result["score"]

        if len(toxicity_above_threshold) > 0:
            logger.warning(f"Detected toxicity in the text: {toxicity_above_threshold}")

            return prompt, False, round(calculate_risk_score(highest_toxicity_score, threshold),2)

        logger.debug(f"Not toxicity found in the text. Results: {results}")

        return prompt, True, round(highest_toxicity_score,2)
