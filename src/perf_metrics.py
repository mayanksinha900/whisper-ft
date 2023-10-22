from typing import Dict

import evaluate
import numpy as np
from attrs import define
from sentence_transformers import SentenceTransformer

from src.text_processor import TextPreprocess


@define
class PerformanceMetrcis:
    labse_path: str
    tp = TextPreprocess()

    def calculate_matching_accuracy(self, sent: str, pred: str) -> float:
        sent_words = self.tp.pipeline(sent).split()
        pred_words = self.tp.pipeline(pred).split()

        min_len = min(len(sent_words), len(pred_words))

        correct_words = 0
        for s, p in zip(sent_words[:min_len], pred_words[:min_len]):
            if s == p:
                correct_words += 1

        acc = float(correct_words) / float(min_len)
        penalty = max(0, len(pred_words) - len(sent_words)) * 0.05
        final_acc = max(0, acc - penalty)

        return round(final_acc * 100, 2)

    def labse_similarity(self, sent: str, pred: str):
        labse = SentenceTransformer(self.labse_path)
        embdeds = labse.encode([sent, pred])
        similarity = np.dot(embdeds[0], embdeds[1]) / (
            np.linalg.norm(embdeds[0]),
            np.linalg.norm(embdeds[1]),
        )

        return round(float(similarity[0]) * 100, 2)

    def bert_score(self, sent: str, pred: str) -> float:
        bert = evaluate.load("bertscore")
        result = bert.compute(
            predictions=[pred], 
            references=[sent], 
            model_type="distilbert-base-uncased"
        )

        f1 = result["f1"][0]

        return round(f1 * 100, 2)

    def inference_score(self, sent: str, pred: str) -> Dict:
        return {
            "match_accuracy": self.calculate_matching_accuracy(sent,pred),
            "labse_accuracy": self.labse_similarity(sent, pred),
            "bert_accuracy": self.bert_score(sent, pred)
        }