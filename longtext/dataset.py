import random

import torch
from loguru import logger
from neetils import read_jsonl
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class LongTextDataset(Dataset):

    def __init__(
        self,
        raw_data: list,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_chunk: int = -1,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chunk = max_chunk
        self.seperator = "[unused1]"
        self.seperator_id = self.tokenizer.convert_tokens_to_ids(self.seperator)
        self.cache_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index: int):
        if index in self.cache_dict:
            return self.cache_dict[index]

        item = self.raw_data[index]
        text = item["text"].replace("\n", self.seperator)

        if text.count(self.seperator) != len(item["label"]):
            logger.error(f"Text {text} has different number of label {item['label']}")
            rnd_idx = random.randint(0, len(self.raw_data) - 1)
            return self.__getitem__(rnd_idx)

        input_ids_list = [[]]
        attention_mask_list = [[]]
        labels_list = [[]]

        for idx, line in enumerate(text.split(self.seperator)):
            if idx == len(text.split(self.seperator)) - 1:
                continue
            inputs = self.tokenizer(line, add_special_tokens=False)
            inputs["input_ids"] += [self.seperator_id]
            inputs["attention_mask"] += [1]

            if len(inputs["input_ids"]) + len(input_ids_list[-1]) + 2 > self.max_length:
                # 1. padding the last chunk
                input_ids_list[-1] = (
                    [self.tokenizer.cls_token_id]
                    + input_ids_list[-1]
                    + [self.tokenizer.sep_token_id]
                )
                input_ids_list[-1] += [0] * (self.max_length - len(input_ids_list[-1]))
                attention_mask_list[-1] += [1, 1] + [0] * (
                    self.max_length - len(attention_mask_list[-1]) - 2
                )
                labels_list[-1] += [-100] * (self.max_length - len(labels_list[-1]))

                # 2. create a new chunk
                input_ids_list.append([])
                attention_mask_list.append([])
                labels_list.append([])

                input_ids_list[-1].extend(inputs["input_ids"])
                attention_mask_list[-1].extend(inputs["attention_mask"])
                labels_list[-1].append(item["label"][idx])
            else:
                input_ids_list[-1].extend(inputs["input_ids"])
                attention_mask_list[-1].extend(inputs["attention_mask"])
                labels_list[-1].append(item["label"][idx])

        # padding the last chunk
        input_ids_list[-1] = (
            [self.tokenizer.cls_token_id]
            + input_ids_list[-1]
            + [self.tokenizer.sep_token_id]
        )
        input_ids_list[-1] += [0] * (self.max_length - len(input_ids_list[-1]))
        attention_mask_list[-1] += [1, 1] + [0] * (
            self.max_length - len(attention_mask_list[-1]) - 2
        )
        labels_list[-1] += [-100] * (self.max_length - len(labels_list[-1]))

        if self.max_chunk > 0:
            input_ids_list = input_ids_list[: self.max_chunk]
            attention_mask_list = attention_mask_list[: self.max_chunk]
            labels_list = labels_list[: self.max_chunk]

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            label=labels_list,
        )


def data_collator(batch):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["label"] for example in batch]

    def to_tensor(data):
        res = []
        for item in data:
            res.extend(item)
        return torch.tensor(res, dtype=torch.long)

    input_ids = to_tensor(input_ids)
    attention_mask = to_tensor(attention_mask)
    labels = to_tensor(labels)

    return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def make_supervisied_data_modules(data_args, tokenizer: PreTrainedTokenizer):
    if data_args.data_path.endswith(".jsonl"):
        raw_data = read_jsonl(data_args.data_path)
    else:
        raise ValueError("Unsupported data format")

    if data_args.eval_data_path.endswith(".jsonl"):
        eval_raw_data = read_jsonl(data_args.eval_data_path)
    else:
        raise ValueError("Unsupported data format")

    logger.info(
        f"Load {len(raw_data)} training samples and {len(eval_raw_data)} eval samples"
    )

    train_dataset = LongTextDataset(raw_data, tokenizer)
    eval_dataset = LongTextDataset(eval_raw_data, tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
