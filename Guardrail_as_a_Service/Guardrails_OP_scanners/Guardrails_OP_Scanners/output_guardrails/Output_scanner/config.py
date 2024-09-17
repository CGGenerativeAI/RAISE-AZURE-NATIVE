import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class ServiceConfig:  

    config_data: dict = field(default_factory= lambda : {"Output":{
                                                        "Relevance": {"threshold": 0.5},
                                                        "Bias": {"threshold": 0.75},
                                                        "BanTopics": {"topics": ["violence","politics"],"threshold": 0.75},
                                                        "Toxicity": {"threshold": 0.75},
                                                        "CustomGuard":{"threshold":0.8}
                                                        }
                                                    })

    prompt: str = "What is life?"
    model_output: str = "life is a Race"
    inputs: dict = field(default_factory= lambda : {"prompt":[ServiceConfig.prompt],"model_output":[ServiceConfig.model_output]})

    








