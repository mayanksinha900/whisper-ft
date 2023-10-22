import faster_whisper
import librosa
import pandas as pd
import torch
import yaml

from src.perf_metrics import PerformanceMetrcis
from src.reduce_noise import ReduceNoise
from src.utils import stt_result

with open("config/inference.yaml") as f:
    configs = yaml.safe_load(f)


model_id = configs["model"]
device = configs["device"]
labse_path = configs["labse"]
compute_type = configs["compute_type"]
audio_path = configs["audio_path"]
sr = configs["sr"]
task = configs["task"]

WHIPSER_SR = 16000

rn = ReduceNoise(sr=WHIPSER_SR, time_mask_smooth_ms=20)
pm = PerformanceMetrcis(labse_path=labse_path)
model = faster_whisper.WhisperModel(model_id, device=device, compute_type=compute_type)
torch.cuda.empty_cache()


def main():
    data = pd.read_csv(f"{audio_path}/metadata.csv", sep="|")
    final_output = {"match_accuracy": 0.0, "labse_accuracy": 0.0, "bert_accuracy": 0.0}

    for _, row in data.iterrows():
        audio = row["audio"]
        text = row["text"]
        y, _ = librosa.load(audio, sr=sr)
        if sr != WHIPSER_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=WHIPSER_SR)
        whisper_output = stt_result(y, model, "en", task=task)
        whisper_text = whisper_output["text"]

        metrics = pm.inference_score(sent=text, pred=whisper_text)

        final_output["match_accuracy"] += metrics["match_accuracy"]
        final_output["labse_accuracy"] += metrics["labse_accuracy"]
        final_output["bert_accuracy"] += metrics["bert_accuracy"]

    match_acc = final_output["match_accuracy"] / len(data)    
    labse_acc = final_output["labse_accuracy"] / len(data)    
    bert_acc = final_output["bert_accuracy"] / len(data)

    print("Match Accuracy: ", match_acc)
    print("LaBSE Accuracy: ", labse_acc)
    print("BERT Accuracy: ", bert_acc)

if __name__=="__main__":
    main()