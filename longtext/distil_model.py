from model import LongTextBertModel


class LongTextDistilBertModel(LongTextBertModel):
    def __init__(self, config):
        super(LongTextDistilBertModel, self).__init__(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs
