import os
from shutil import copyfile
from typing import Optional

import yaml
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

from src.trainer import load_configs


class ModelMerge:
    def __init__(self, config:Optional[str] = None):
        if config is None:
            config = load_configs()
        else:
            with open("./config/peft_merge_only.yaml", "r") as file:
                config = yaml.safe_load(file)
        print(config)
        if isinstance(config, dict):
            self.model_path = config["model_config"]["model_path"]
            self.model_name = config["model_config"]["model_name"]
            self.output = config["training_params"]["output_model"]
        else: 
            self.model_path = config.model_config.model_path
            self.model_name = config.model_config.model_name
            self.output = config.training_params.output_model
            
        if self.model_name == "" and self.model_path == "":

            raise Exception("Need to specify either model name or path.")
        else:
            self.model_id = (
                self.model_name if self.model_path == "" else self.model_path
            )

    def merge(self):
        save_path = f"models/{self.output}/final_model"
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        model = PeftModel.from_pretrained(model, f"models/{self.output}/adapter_model")
        model.config.use_cache = True
        m1 = model.merge_and_unload()
        m1.save_pretrained(save_path)
        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        if os.path.isfile(tokenizer_file):
            copyfile(tokenizer_file, f"{save_path}/tokenizer.json")
        else:
            copyfile("tokenizer.json", f"{save_path}/tokenizer.json")