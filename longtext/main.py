import json
import pathlib
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed
import transformers
from common import init_logger, launch_debugger
from dataset import make_supervisied_data_modules
from distil_model import LongTextDistilBertModel
from evaluation import eval_model
from loguru import logger
from model import LongTextBertModel
from trainer import LongTextTrainer
from transformers import BertConfig, BertTokenizer, HfArgumentParser

MODEL_CLASS_MAP = {
    "bert": (LongTextBertModel, BertConfig, BertTokenizer),
    "distil_bert": (LongTextDistilBertModel, BertConfig, BertTokenizer),
}


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    pretrain_learning_rate: float = (
        5e-5  # the learning rate for pretrain model, bert etc.
    )
    distil_temperature: float = 1.0  # the temperature of distillation model
    do_distil: bool = False  # whether to distil the model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = "bert-base-chinese"
    model_type: str = "bert"
    teacher_model_type: str = "bert"  # bert or distilbert
    num_distil_layers: int = 6  # layer number of distillation model
    teacher_model_path: str = None  # path of teacher model


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = "train.jsonl"
    eval_data_path: str = "eval.jsonl"
    max_length: int = 512
    bert_debug: bool = False


def load_teacher_model(model_args, data_args):
    model_class, config_class, tokenizer_class = MODEL_CLASS_MAP[
        model_args.teacher_model_type
    ]
    config = config_class.from_pretrained(model_args.teacher_model_path)

    tokenizer = tokenizer_class.from_pretrained(model_args.teacher_model_path)

    model = model_class.from_pretrained(model_args.teacher_model_path, config=config)
    model.tokenizer = tokenizer

    model.eval()
    # freeze teacher model
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Load teacher model from %s" % model_args.teacher_model_path)

    model.eval()
    model.cuda()

    return model


@logger.catch
def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    init_logger(training_args)

    model_class, config_class, tokenizer_class = MODEL_CLASS_MAP[model_args.model_type]

    if data_args.bert_debug:
        launch_debugger()

    logger.info("Model parameters %s" % model_args)
    logger.info("Data parameters %s" % data_args)

    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)

    config = config_class.from_pretrained(model_args.model_name_or_path)
    config.seperator_token_id = tokenizer.convert_tokens_to_ids("[unused1]")

    if training_args.do_distil:
        config.num_hidden_layers = model_args.num_distil_layers
        config.temperature = training_args.distil_temperature

    model = model_class.from_pretrained(model_args.model_name_or_path, config=config)

    model.tokenizer = tokenizer

    data_modules = make_supervisied_data_modules(data_args, tokenizer)

    torch.distributed.barrier()

    if training_args.do_train:
        logger.info("======= Start training =======")

        teacher_model = None
        if model_args.teacher_model_path is not None:
            teacher_model = load_teacher_model(model_args, data_args)

            if len(teacher_model.tokenizer) != len(model.tokenizer):
                logger.warning(
                    "Teacher and student tokenizer not match, teacher: %s, student: %s, attempting to resize..."
                    % (len(teacher_model.tokenizer), len(model.tokenizer))
                )
                model.resize_token_embeddings(len(teacher_model.tokenizer))

        trainer = LongTextTrainer(
            teacher_model=teacher_model,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_modules
        )

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logger.info("Resuming from checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        trainer.save_model()
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        trainer.save_state()
        logger.info("Model saved to %s" % training_args.output_dir)

    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("======= Start predicting =======")
        eval_model(model, data_modules["eval_dataset"])


if __name__ == "__main__":
    train()
