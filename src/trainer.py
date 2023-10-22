from typing import Any, Dict, List, Union
import os
import datasets
import evaluate
import torch
import yaml
from attrs import define
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.config import WhisperConfig
from src.create_data import CreateDataset


def load_configs() -> WhisperConfig:
    with open("config/whisper.yaml", "r") as file:
        config = yaml.safe_load(file)
    return WhisperConfig.validate(config)


@define
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


class Trainer:
    def __init__(self):
        config = load_configs()
        self.data = config.data
        self.model_config = config.model_config
        self.model_params = config.model_params
        self.training_params = config.training_params
        self.pb_api = config.push_bullet_api
        self.dataset = self._load_dataset()

        if self.model_config.model_name == "" and self.model_config.model_path == "":
            raise Exception("Need to specify either model name or path.")
        else:
            self.model_id = (
                self.model_config.model_name
                if self.model_config.model_path == ""
                else self.model_config.model_path
            )

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.model_id,
            language=self.model_config.language,
            task=self.model_config.task,
        )
        self.processor = WhisperProcessor.from_pretrained(
            self.model_id,
            language=self.model_config.language,
            task=self.model_config.task,
        )
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor
        )
        self.metric = evaluate.load(self.model_params.evaluate_method)

        if not os.path.exists(f"models/{self.training_params.output_model}"):
            os.mkdir(f"models/{self.training_params.output_model}")

    def _load_dataset(self) -> datasets.Dataset:
        cd = CreateDataset(self.data)
        return cd.get_dataset()

    def _process_audio(self):
        self.dataset = self.dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=16000)
        )

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def make_inputs_require_grad(self, module, input, output):
        output.requires_grad_(True)

    def train(self):
        print("Data processing started.")
        self._process_audio()
        print("Data processing ended.")
        print("Started feature extraction.")
        dataset = self.dataset.map(
            self.prepare_dataset,
            remove_columns=self.dataset.column_names,
            num_proc=self.training_params.num_proc,
        )
        print("Feature extraction ended.")
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, load_in_8bit=True, device_map=self.model_params.device_map
        )
        model = prepare_model_for_int8_training(model)
        model.model.encoder.conv1.register_forward_hook(self.make_inputs_require_grad)

        config = LoraConfig(
            r=self.model_params.r,
            lora_alpha=self.model_params.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.model_params.lora_dropout,
            bias=self.model_params.bias,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"models/{self.training_params.output_model}",
            per_device_train_batch_size=self.training_params.batch_size,
            gradient_accumulation_steps=self.training_params.gradient_accumulation_steps,
            learning_rate=self.training_params.learning_rate,
            warmup_steps=self.training_params.warmup_steps,
            num_train_epochs=self.training_params.num_train_epochs,
            evaluation_strategy=self.training_params.evaluation_strategy,
            fp16=self.training_params.fp16,
            save_strategy=self.training_params.save_strategy,
            save_steps=self.training_params.save_steps,
            per_device_eval_batch_size=self.training_params.per_device_eval_batch_size,
            generation_max_length=self.training_params.generation_max_length,
            logging_steps=self.training_params.logging_steps,
            max_steps=self.training_params.max_steps,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=self.data_collator,
            # compute_metrics=compute_metrics,
            tokenizer=self.processor.feature_extractor,
            callbacks=[SavePeftModelCallback],
        )

        model.config.use_cache = False
        trainer.train()
        model.save_pretrained(
            f"models/{self.training_params.output_model}/adapter_model"
        )
