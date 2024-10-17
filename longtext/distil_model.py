from model import LongTextBertModel
import torch


class LongTextDistilBertModel(LongTextBertModel):
    def __init__(self, config):
        super(LongTextDistilBertModel, self).__init__(config)

    def forward(
        self, input_ids, attention_mask=None, labels=None, teacher_outputs=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            output_hidden_states=True,
        )

        hidden_states = outputs[0]

        feature = self.create_feature(input_ids, hidden_states)
        logits = self.classifier(feature)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

            loss = ce_loss_fct(
                logits,
                labels[labels != -100],
            )

            if teacher_outputs is not None:
                mse_loss_fct = torch.nn.MSELoss()

                teacher_logits = teacher_outputs[1]
                logits_loss = mse_loss_fct(logits, teacher_logits)
                loss += logits_loss

                teacher_hidden_states = teacher_outputs[2]
                hidden_states = outputs[1]

                for idx, hidden_state in enumerate(hidden_states):
                    hidden_state_loss = mse_loss_fct(
                        hidden_state, teacher_hidden_states[idx * 2]
                    )
                    loss += hidden_state_loss

            outputs = (loss,) + outputs

        return outputs
