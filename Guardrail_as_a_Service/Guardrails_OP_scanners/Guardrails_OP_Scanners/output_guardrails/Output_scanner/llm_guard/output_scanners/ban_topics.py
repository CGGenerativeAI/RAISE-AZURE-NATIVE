from typing import Dict, List, Optional, Any

from llm_guard.input_scanners.ban_topics import BanTopics as InputBanTopics

from .base import Scanner


class BanTopics(Scanner):
    """
    A text scanner that checks whether the generated text output includes banned topics.

    The class uses the zero-shot-classification model from Hugging Face to scan the topics present in the text.
    """

    def __init__(
        self,
        model: Optional[Dict] = None,
        use_onnx: bool = False,
    ):
        """
        Initializes BanTopics with a list of topics and a probability threshold.

        Parameters:
            model (Dict, optional): The name of the zero-shot-classification model to be used. Default is MODEL_BASE.
            use_onnx (bool, optional): Whether to use ONNX for inference. Default is False.

        Raises:
            ValueError: If no topics are provided.
        """
        self._scanner = InputBanTopics(
            model=model, use_onnx=use_onnx
        )

    def scan(self, prompt: str, output: str,**kwargs: Optional[Dict[str, Any]]) -> (str, bool, float):
        return self._scanner.scan(output,**kwargs)
