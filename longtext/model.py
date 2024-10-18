import torch
from loguru import logger
from transformers import BertConfig, BertModel, BertPreTrainedModel, PreTrainedTokenizer


class LongTextBertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(LongTextBertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.sep_classifier = torch.nn.Linear(config.hidden_size, 2)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size * 2, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.init_weights()

    def create_feature(self, input_ids, hidden_states):
        cls_feature = hidden_states[:, 0, :]
        sep_feature = hidden_states[input_ids == self.tokenizer.sep_token_id]

        global_feature = torch.max(
            torch.sub(cls_feature, sep_feature), dim=0, keepdim=False
        )[0]
        line_feature = hidden_states[input_ids == self.config.seperator_token_id]
        global_feature = global_feature.repeat(line_feature.size(0), 1)

        concat_feature = torch.cat([global_feature, line_feature], dim=-1)
        feature = self.ffn(concat_feature)

        return feature

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            output_hidden_states=True,
        )

        hidden_states = outputs[0]

        feature = self.create_feature(input_ids, hidden_states)
        logits = self.sep_classifier(feature)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

            loss = loss_fct(
                logits,
                labels[labels != -100],
            )

            outputs = (loss,) + outputs

        return outputs

    def predict(self, text: str, max_length: int = 512):
        seperator = "\n"
        if not text.endswith(seperator):
            text += seperator

        input_ids_list = [[]]
        attention_mask_list = [[]]

        for idx, line in enumerate(text.split(seperator)):
            if idx == len(text.split(seperator)) - 1:
                continue
            inputs = self.tokenizer(line, add_special_tokens=False)
            inputs["input_ids"] += [self.config.seperator_token_id]
            inputs["attention_mask"] += [1]

            if len(inputs["input_ids"]) + len(input_ids_list[-1]) + 2 > max_length:
                # 1. padding the last chunk
                input_ids_list[-1] = (
                    [self.tokenizer.cls_token_id]
                    + input_ids_list[-1]
                    + [self.tokenizer.sep_token_id]
                )
                input_ids_list[-1] += [0] * (max_length - len(input_ids_list[-1]))
                attention_mask_list[-1] += [1, 1] + [0] * (
                    max_length - len(attention_mask_list[-1]) - 2
                )

                # 2. create a new chunk
                input_ids_list.append([])
                attention_mask_list.append([])

                input_ids_list[-1].extend(inputs["input_ids"])
                attention_mask_list[-1].extend(inputs["attention_mask"])
            else:
                input_ids_list[-1].extend(inputs["input_ids"])
                attention_mask_list[-1].extend(inputs["attention_mask"])

        # padding the last chunk
        input_ids_list[-1] = (
            [self.tokenizer.cls_token_id]
            + input_ids_list[-1]
            + [self.tokenizer.sep_token_id]
        )
        input_ids_list[-1] += [0] * (max_length - len(input_ids_list[-1]))
        attention_mask_list[-1] += [1, 1] + [0] * (
            max_length - len(attention_mask_list[-1]) - 2
        )

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            attention_mask_list, dtype=torch.long, device=self.device
        )

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs[0]
        pred = torch.softmax(logits, dim=-1)
        pred = torch.argmax(pred, dim=-1)
        pred = pred.cpu().tolist()
        logger.debug(pred)

        results = [""]
        for idx, line in enumerate(text.split(seperator)):
            if idx == len(text.split(seperator)) - 1:
                continue
            
            if pred[idx] == 0:
                results[-1] += line + seperator
            else:
                results[-1] += line + seperator
                results.append("")
        results = [r.strip() for r in results if r.strip()]
        return results

