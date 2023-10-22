from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
    model_path: str
    task: str
    language: str

class ModelParams(BaseModel):
    evaluate_method: str
    device_map: str
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str

class TrainingParams(BaseModel):
    output_model: str
    num_proc: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    num_train_epochs: int
    evaluation_strategy: str
    fp16: bool
    save_strategy: str
    save_steps: int
    per_device_eval_batch_size: int
    generation_max_length: int
    logging_steps: int
    max_steps: int
    split_size: float


class WhisperConfig(BaseModel):
    model_config:ModelConfig
    model_params:ModelParams
    data:str
    training_params:TrainingParams
    push_bullet_api:str
