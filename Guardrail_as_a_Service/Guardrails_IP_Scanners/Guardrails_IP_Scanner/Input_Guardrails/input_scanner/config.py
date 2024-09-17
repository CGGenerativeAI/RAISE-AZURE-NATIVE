import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class ServiceConfig:  
 
    config_data: dict = field(default_factory= lambda : {"Input":{
                                                            "Toxicity": {"threshold": 0.7},
                                                            "BanTopics": {"topics": ["violence","politics"],"threshold": 0.6},
                                                            "PromptInjection": {"threshold": 0.75},
                                                            "CustomGuard":{"threshold":0.8}
                                                            }
                                                        })

    prompt: str = "What is life?"
    inputs: dict = field(default_factory= lambda : {"prompt":[ServiceConfig.prompt]})
  






