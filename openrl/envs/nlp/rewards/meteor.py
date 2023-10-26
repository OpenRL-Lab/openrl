from pathlib import Path
from typing import Any, Dict

import evaluate

import openrl.envs.nlp as nlp


class VirtualMetric:
    def compute(self, predictions: Any, references: Any) -> Dict[str, float]:
        return {"meteor": 0.0}


class Meteor:
    def __init__(self, meteor_coeff: int, test: bool = False) -> None:
        super().__init__()
        self._meteor_coeff = meteor_coeff
        if test:
            self._metric = VirtualMetric()
        else:
            self._metric = evaluate.load(
                str(Path(nlp.__file__).parent / "utils/metrics/meteor.py")
            )

    def __call__(
        self,
        data: Dict[str, Any],
    ):
        generated_texts = [data["generated_texts"]]
        reference_texts = [data["reference_texts"]]
        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]

        reward = score * self._meteor_coeff
        info = {"meteor": score}

        return reward, info
