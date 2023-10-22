import pandas as pd
import datasets
import librosa
import wave
import os
import sys

class CreateDataset:
    def __init__(self, path: str):
        self.root_folder = f"data/{path}"
        try:
            self.df = pd.read_csv(f"{self.root_folder}/metadata.csv", sep="|")
            self.df.columns = ["path", "sentence"]

            with wave.open(self.df.at[0, "path"], "rb") as fh:
                (_, _, srate, _, _, _) = fh.getparams()
            self.sr = srate
        except Exception as err:
            print(f"Error occured: {err}")
            print("Couldn't find the metadata.csv.")
            print("Assuming existence of datasets folder. Loading default SR = 24k")
            self.df = None
            self.sr = 24000

    def get_audio_array(self,path):
        y, _ = librosa.load(path, sr=self.sr)
        return y


    def get_dataset(self) -> datasets.Dataset:
        dataset_path = f"{self.root_folder}/datasets"
        if os.path.exists(dataset_path):
            return datasets.Dataset.load_from_disk(dataset_path)
        else:
            if self.df is None:
                print("No data available for training. Exiting...")
                sys.exit()

        self.df["array"] = self.df["path"].apply(self.get_audio_array)

        audio = []
        sentence  = []

        for _, row in self.df.iterrows():
            aud = {
                "array": row["array"],
                "path": row["path"],
                "sampling_rate": self.sr
            }
            sent = row["sentence"]

            audio.append(aud)
            sentence.append(sent)

        del self.df

        final_dict = {
            "audio": audio,
            "sentence": sentence
        }

        dataset = datasets.Dataset.from_dict(final_dict)
        dataset.save_to_disk(dataset_path=dataset_path)

        return dataset

