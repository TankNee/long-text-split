import gradio as gr
from model import LongTextBertModel
import torch
from transformers import AutoTokenizer
from neetils import read_jsonl
import random
from loguru import logger


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", "-m", type=str, default="longtext/model")
    parser.add_argument("--test_case", "-tc", type=str, default="test_v2.jsonl")

    return parser.parse_args()

def main():
    args = get_args()
    model: LongTextBertModel = LongTextBertModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.tokenizer = tokenizer

    def predict(text):
        results = model.predict(text)
        logger.debug(results)
        return "\n\n----------------\n\n".join(results)
    
    test_cases = read_jsonl(args.test_case)
    test_cases = [t["text"] for t in test_cases]
    examples = random.sample(test_cases, 10)
    demo = gr.Interface(
        fn=predict,
        inputs=["text"],
        outputs=["text"],
        examples=examples
    )

    demo.launch()

if __name__ == "__main__":
    main()

