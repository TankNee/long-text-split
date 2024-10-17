import torch
from dataset import data_collator
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger


def eval_model(model, eval_dataset):

    data_loader = DataLoader(
        dataset=eval_dataset, batch_size=2, collate_fn=data_collator, shuffle=True
    )

    y_true = []
    y_pred = []
    for batch in tqdm(data_loader, desc="Evaluating"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)

        y_true.extend(batch["labels"][batch["labels"] != -100].cpu().tolist())
        logits = outputs[-1]
        pred = torch.softmax(logits, dim=-1)
        pred = torch.argmax(pred, dim=-1)
        y_pred.extend(pred.cpu().tolist())

    logger.info(f"Report: \n {classification_report(y_true, y_pred, digits=4)}")
