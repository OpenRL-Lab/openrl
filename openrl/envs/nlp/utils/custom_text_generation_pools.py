from openrl.envs.nlp.utils.text_generation_pool import Sample, TextGenPool
from openrl.supports.opendata.utils.opendata_utils import load_dataset


class CommonGen(TextGenPool):
    @classmethod
    def prepare(
        cls,
        split: str,
        concept_separator_token: str = " ",
        concept_end_token=" ",
        prefix: str = "summarize: ",
    ) -> "TextGenPool":
        ds = load_dataset("gem", "common_gen")
        samples = []
        split_id = CommonGen.gen_split_name(split)
        for ix, item in enumerate(ds[split_id]):
            concepts = concept_separator_token.join(item["concepts"])
            concepts = prefix + concepts
            concepts += concept_end_token
            if item["target"] == "":
                # just to avoid breaking of metric computation
                item["target"] = "empty reference"
            targets = [item["target"]]
            sample = Sample(
                id=f"{split}_{ix}",
                prompt_or_input_text=concepts,
                references=targets,
                meta_data={"concepts": item["concepts"]},
            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class DailyDialog(TextGenPool):
    EOU_TOKEN = "<EOU>"

    @classmethod
    def prepare(
        cls, data_path: str, split: str, context_size: int, small_debug: bool = False
    ):
        split = CommonGen.gen_split_name(split)
        dataset = load_dataset(data_path, split)

        samples = []
        utterance_id = 0
        for item in dataset:
            contexts = []
            for utterance, emotion, intent in zip(
                item["dialog"], item["emotion"], item["act"]
            ):
                if len(contexts) >= context_size:
                    context = DailyDialog.EOU_TOKEN.join(contexts[-context_size:])
                    context += " " + DailyDialog.EOU_TOKEN
                    target = utterance + DailyDialog.EOU_TOKEN

                    sample = Sample(
                        id=utterance_id,
                        prompt_or_input_text=context,
                        references=[target],
                        meta_data={"emotion": [emotion], "intent": [intent]},
                    )
                    samples.append(sample)
                contexts.append(utterance)
                utterance_id += 1

        dp_instance = cls(samples)
        return dp_instance
