from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import Dataset, concatenate_datasets


import json
import argparse
import os
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
MODEL_DIR = os.getenv("MODEL_PATH")
CACHE_PATH = os.getenv("CACHE_PATH")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
TRAIN_PATH = os.getenv("TRAIN_PATH")
TEST_PATH = os.getenv("TEST_PATH")


def soft_cross_entropy(preds, soft_targets):
    log_probs = F.log_softmax(preds, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()  # Weight with noise


# Helper functions for setting up distributed training
def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=3600))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_dpp():
    dist.destroy_process_group()


def merge_labels(example):
    return {"labels": [example["label_0"], example["label_1"]]}


def main(
    train_split,
    test_split,
    val_split,
    num_epochs,
    accumulation_steps,
    model_output_dir,
    batch_size,
    weight_decay,
    learning_rate,
    use_snorkel_labels,
):
    local_rank = setup_ddp()
    rank = dist.get_rank()
    # Settings
    # Load tokenized data
    train_dataset = Dataset.from_parquet(TRAIN_PATH, cache_dir=CACHE_PATH)
    train_dataset = train_dataset.remove_columns(["content", "label_0", "label_1"])
    if use_snorkel_labels:
        train_dataset = train_dataset.map(merge_labels)
        train_dataset = train_dataset.remove_columns(
            column_names=["label_0", "label_1"]
        )
    else:
        train_dataset = train_dataset.remove_columns(["content", "label_0", "label_1"])
        train_dataset = train_dataset.map(lambda x: {"labels": [1.0, 0.0]})
    train_dataset = train_dataset.shuffle(seed=42)

    # Load positive data
    test_dataset = Dataset.from_parquet(
        TEST_PATH, cache_dir=CACHE_PATH
    )
    test_dataset = test_dataset.remove_columns(
        column_names=["std_message_id", "speaker", "timestamp", "subject", "content"]
    )
    test_dataset = test_dataset.map(lambda x: {"labels": [0.0, 1.0]})
    test_dataset = test_dataset.shuffle(seed=42)

    unlabeled_size = int(len(train_dataset))
    labeled_size = int(len(test_dataset))

    train_size_unlabeled = int(unlabeled_size * train_split)
    train_size_labeled = int(labeled_size * train_split)

    val_size_unlabeled = int(unlabeled_size * val_split)
    val_size_labeled = int(labeled_size * val_split)

    train_unlabeled_idxs = list(range(train_size_unlabeled))
    val_unlabeled_idxs = list(
        range(train_size_unlabeled, train_size_unlabeled + val_size_unlabeled)
    )
    test_unlabeled_idxs = list(
        range(train_size_unlabeled + val_size_unlabeled, unlabeled_size)
    )

    train_labeled_idxs = list(range(train_size_labeled))
    val_labeled_idxs = list(
        range(train_size_labeled, train_size_labeled + val_size_labeled)
    )
    test_labeled_idxs = list(range(train_size_labeled + val_size_labeled, labeled_size))

    train_unlabeled = train_dataset.select(train_unlabeled_idxs)
    train_labeled = test_dataset.select(train_labeled_idxs)
    final_train_dataset = concatenate_datasets([train_unlabeled, train_labeled])
    final_train_dataset = final_train_dataset.shuffle(seed=42)

    val_unlabeled = train_dataset.select(val_unlabeled_idxs)
    val_labeled = test_dataset.select(val_labeled_idxs)
    final_val_dataset = concatenate_datasets([val_unlabeled, val_labeled])
    final_val_dataset = final_val_dataset.shuffle(seed=42)

    test_unlabeled = train_dataset.select(test_unlabeled_idxs)
    test_labeled = test_dataset.select(test_labeled_idxs)
    final_test_dataset = concatenate_datasets([test_unlabeled, test_labeled])
    final_test_dataset = final_test_dataset.shuffle(seed=42)

    if rank == 0:
        print(final_train_dataset.column_names)
        print(final_val_dataset.column_names)
        print(final_test_dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_sampler = DistributedSampler(
        final_train_dataset, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(final_val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        final_train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = DataLoader(
        final_val_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=val_sampler,
        drop_last=False,
    )

    test_loader = DataLoader(
        final_test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_last=False,
    )
    # Device setup
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        output_dir = f"{OUTPUT_PATH}/output_{MODEL_NAME}_{EXPERIMENT_NAME}"
        logging_dir = f"{OUTPUT_PATH}/logs_{MODEL_NAME}_{EXPERIMENT_NAME}"
        os.makedirs(
            output_dir, exist_ok=True
        )  # Create output directory if it doesn't exist
        os.makedirs(
            logging_dir, exist_ok=True
        )  # Create logging directory if it doesn't exist

    # Model, optimizer, scaler
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = GradScaler("cuda", enabled=True)
    # Training loop
    best_val = float("inf")
    training = True
    if training:
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loader.sampler.set_epoch(epoch)
            optimizer.zero_grad()
            avg_training_loss = 0.0
            total_samples = 0
            for step, batch in tqdm(
                enumerate(train_loader, 1), total=len(train_loader)
            ):
                # Move inputs & labels to device
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                soft_labels = batch["labels"].to(device)  # shape (B,2)

                with autocast(device_type="cuda"):
                    logits = model(**inputs).logits  # (B,2)
                    loss = soft_cross_entropy(logits, soft_labels)

                # Backpropagate
                scaler.scale(loss).backward()

                avg_training_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

                # Step optimizer
                if step % accumulation_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            if rank == 0:
                print("synchronized after training, computing loss")
            loss_tensor = torch.tensor(avg_training_loss, device=device)
            count_tensor = torch.tensor(total_samples, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            global_avg_train_loss = (loss_tensor / count_tensor).item()
            if rank == 0:
                print(f"Train loss: {global_avg_train_loss}, Reached validation")
            model.eval()
            val_loader.sampler.set_epoch(epoch)
            avg_val_loss = 0.0
            total_val_samples = 0
            true_positives = 0
            total_positives = 0
            predicted_positives = 0
            total_preds = 0
            with torch.no_grad():
                for step, batch in tqdm(
                    enumerate(val_loader, 1), total=len(val_loader)
                ):
                    inputs = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                    }
                    labels = batch["labels"][:, 1].long()
                    with autocast(device_type="cuda"):
                        outputs = model(**inputs).logits
                        loss = soft_cross_entropy(logits, soft_labels)
                    avg_val_loss += loss.item() * batch["input_ids"].size(0)
                    total_val_samples += batch["input_ids"].size(0)

                    preds = torch.argmax(outputs, dim=1)
                    positive_mask = labels == 1
                    true_positives += (preds[positive_mask] == 1).sum().item()
                    total_positives += positive_mask.sum().item()
                    predicted_positives += (preds == 1).sum().item()
                    total_preds += len(preds)
                dist.barrier()
                if rank == 0:
                    print("synchronized after validation, computing stats")
                stats_tensor = torch.tensor(
                    [
                        avg_val_loss,
                        total_val_samples,
                        true_positives,
                        total_positives,
                        predicted_positives,
                        total_preds,
                    ],
                    device=device,
                )
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                (
                    avg_val_loss,
                    total_val_samples,
                    true_positives,
                    total_positives,
                    predicted_positives,
                    total_preds,
                ) = stats_tensor.tolist()
                global_val_loss = avg_val_loss / total_val_samples
                val_stats = {
                    "avg_training_loss": global_avg_train_loss,
                    "avg_val_loss": global_val_loss,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "true_positives": true_positives,
                    "total_positives": total_positives,
                    "predicted_positives": predicted_positives,
                    "total_test": int(len(final_val_dataset)),
                    "total_preds": total_preds,
                }
                if rank == 0:
                    with open(f"{logging_dir}/prevalence_val_{epoch}.json", "w") as f:
                        json.dump(val_stats, f, indent=2)
                        print(
                            f"Val stats saved \
                                    to '{logging_dir}/prevalence_val_{epoch}.json'."
                        )
                    state_dict = model.module.state_dict()
                    torch.save(
                        state_dict,
                        os.path.join(output_dir, f"prevalence_model_{epoch}.pth"),
                    )
                    if global_val_loss < best_val:
                        torch.save(
                            state_dict,
                            os.path.join(output_dir, "best_prevalence_model.pth"),
                        )
                        best_val = global_val_loss
                    print(f"Epoch {epoch} complete.")
                dist.barrier()
                print(f"Synchronized after completing epoch {epoch}")

    if rank == 0:
        print("Reached testing")
        best_path = os.path.join(output_dir, "best_prevalence_model.pth")
        model.module.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        true_positives = 0
        total_positives = 0
        predicted_positives = 0
        total_preds = 0

        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }

                labels = batch["labels"][:, 1].long()
                with autocast(device_type="cuda"):
                    outputs = model(**inputs).logits

                preds = torch.argmax(outputs, dim=1)
                positive_mask = labels == 1
                true_positives += (preds[positive_mask] == 1).sum().item()
                total_positives += positive_mask.sum().item()
                predicted_positives += (preds == 1).sum().item()
                total_preds += len(preds)

        testing_stats = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "true_positives": true_positives,
            "total_positives": total_positives,
            "predicted_positives": predicted_positives,
            "total_test": int(len(final_test_dataset)),
            "total_preds": total_preds,
        }

        with open(f"{logging_dir}/prevalence_stats_test.json", "w") as f:
            json.dump(testing_stats, f, indent=2)
            print(
                f"Updated testing stats saved \
                        to '{logging_dir}/prevalence_stats_test.json'."
            )

        print("Testing complete.")
    dist.barrier()
    if rank == 0:
        print("Synchronized after Testing complete, cleaning up")
    cleanup_dpp()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train and test snorkel prediction model"
    )
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--use_snorkel_labels", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = main(
        train_split=args.train_split,
        test_split=args.test_split,
        val_split=args.val_split,
        num_epochs=args.num_epochs,
        accumulation_steps=args.accumulation_steps,
        model_output_dir=args.model_output_dir,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        use_snorkel_labels=args.use_snorkel_labels,
    )
