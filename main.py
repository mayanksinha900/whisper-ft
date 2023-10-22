import os
from argparse import ArgumentParser

from src.peft_merge import ModelMerge
from src.trainer import Trainer

ap = ArgumentParser()


def train():
    if not os.path.exists("models/"):
        os.mkdir("models")
    trainer = Trainer()
    trainer.train()
    mm = ModelMerge()
    mm.merge()
    output_path = trainer.training_params.output_model
    with open("tests/tmp.txt", "w") as f:
        f.write(output_path)
    return None


def merge():
    mm = ModelMerge("config/peft_merger_only.yaml")
    mm.merge()
    output_path = mm.output
    with open("tests/tmp.txt", "w") as f:
        f.write(output_path)
    return None


if __name__ == "__main__":
    ap.add_argument("-t", "--type", default="train")
    args = ap.parse_args()
    if args.type == "train":
        train()
    elif args.type == "merge":
        merge()
    else:
        raise Exception("Wrong type parsed.")
