from typing import Dict, List, Optional, Any
from llm_guard.input_scanners.custom_guard import CustomGuard as InputCustomGuard

from .base import Scanner


class CustomGuard(Scanner):
    """
    A scanner that detects code snippets in the model output and blocks them.
    """

    def __init__(self, model: Optional[Dict] = None,
                 use_onnx: bool = False):
        """
        Initialize a new CustomGuard scanner.

        Parameters:
            use_onnx: Whether to use the ONNX model for scanning.
        """

        self._scanner = InputCustomGuard(model = model, use_onnx = use_onnx)

    
    def scan(self, prompt: str, output: str, **kwargs: Optional[Dict[str, Any]]) -> (str, bool, float):
        return self._scanner.scan(output,**kwargs)