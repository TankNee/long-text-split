import torch

# from flash_attn.models.bert import BertModel
from transformers import BertConfig, BertModel, BertPreTrainedModel


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
