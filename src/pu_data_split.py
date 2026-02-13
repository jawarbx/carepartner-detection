import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

tqdm.pandas()

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
TRAIN_PATH = os.getenv("TRAIN_PATH")
TEST_PATH = os.getenv("TEST_PATH")
DEMO_PATH = os.getenv("DEMO_PATH")
CACHE_PATH = os.getenv("CACHE_PATH")
MSG_TO_ENC = os.getenv("MSG_TO_ENC")
MSG_TO_PAT = os.getenv("MSG_TO_PAT")
NONCARETAKER_PATH = os.getenv("NONCARETAKER_PATH")
CARETAKER_PATH = os.getenv("CARETAKER_PATH")

MAX_LEN = 512

if not all(
    [
        MODEL_NAME,
        DATASET_PATH,
        OUTPUT_PATH,
        MODEL_PATH,
        TRAIN_PATH,
        TEST_PATH,
        DEMO_PATH,
        CACHE_PATH,
        MSG_TO_ENC,
        MSG_TO_PAT,
        NONCARETAKER_PATH,
        CARETAKER_PATH,
    ]
):
    missing = [
        var
        for var, val in {
            "MODEL_NAME": MODEL_NAME,
            "DATASET_PATH": DATASET_PATH,
            "OUTPUT_PATH": OUTPUT_PATH,
            "MODEL_PATH": MODEL_PATH,
            "TRAIN_PATH": TRAIN_PATH,
            "TEST_PATH": TEST_PATH,
            "DEMO_PATH": DEMO_PATH,
            "CACHE_PATH": CACHE_PATH,
            "MSG_TO_ENC": MSG_TO_ENC,
            "MSG_TO_PAT": MSG_TO_PAT,
            "NONCARETAKER_PATH": NONCARETAKER_PATH,
            "CARETAKER_PATH": CARETAKER_PATH,
        }.items()
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def merge_labels(example):
    return {"labels": [example["label_0"], example["label_1"]]}


class Positive_Unlabeled_Data_Split:
    def __init__(
        self,
        train_split=0.75,
        test_split=0.05,
        seed=42,
        batch_size=512,
        model_name=MODEL_NAME,
    ):
        temp = pd.read_csv(MSG_TO_ENC)
        temp_2 = pd.read_csv(MSG_TO_PAT)
        demos = pd.read_csv(DEMO_PATH)
        df_merged_nc = pd.read_csv(NONCARETAKER_PATH)
        df_merged_nc.std_message_id = df_merged_nc.std_message_id.astype(int)
        df_merged_nc = df_merged_nc.merge(temp, on="std_message_id", how="left")
        df_merged_nc = df_merged_nc.merge(temp_2, on="std_message_id", how="left")
        df_merged_nc = df_merged_nc.merge(demos, on="pat_owner_id", how="left")

        df_merged_nc.timestamp = df_merged_nc.timestamp.progress_apply(
            lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        )
        df_merged_nc.dob = df_merged_nc.dob.progress_apply(
            lambda s: (
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                if s != "Unknown" and isinstance(s, str)
                else s
            )
        )
        df_merged_nc = df_merged_nc[
            ["content", "std_message_id", "speaker", "timestamp", "subject"]
        ]
        df_merged_nc["timestamp"] = df_merged_nc["timestamp"].astype(str)
        cols = list(df_merged_nc.columns)[1:]

        train_path = f"{OUTPUT_PATH}/{TRAIN_PATH}"
        train_dataset = pd.read_parquet(train_path)

        mask = train_dataset["content"] == df_merged_nc["content"]
        train_dataset[cols] = df_merged_nc[cols].where(mask)
        train_dataset = Dataset.from_pandas(train_dataset)
        train_dataset = train_dataset.map(merge_labels)
        train_dataset = train_dataset.remove_columns(
            ["label_0", "label_1", "speaker", "timestamp", "subject"]
        )
        train_dataset = train_dataset.shuffle(seed=seed)

        # Load positive data
        test_dataset = Dataset.from_parquet(
            TEST_PATH,
            cache_dir=CACHE_PATH,
        )
        test_dataset = test_dataset.remove_columns(
            column_names=["speaker", "timestamp", "subject"]
        )
        test_dataset = test_dataset.map(lambda x: {"labels": [0.0, 1.0]})
        test_dataset = test_dataset.shuffle(seed=seed)

        unlabeled_size = int(len(train_dataset))
        labeled_size = int(len(test_dataset))

        train_size_unlabeled = int(unlabeled_size * train_split)
        train_size_labeled = int(labeled_size * train_split)

        val_size_unlabeled = int(unlabeled_size * test_split)
        val_size_labeled = int(labeled_size * test_split)

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
        test_labeled_idxs = list(
            range(train_size_labeled + val_size_labeled, labeled_size)
        )

        train_unlabeled = train_dataset.select(train_unlabeled_idxs)
        train_labeled = test_dataset.select(train_labeled_idxs)
        final_train_dataset = concatenate_datasets([train_unlabeled, train_labeled])
        final_train_dataset = final_train_dataset.shuffle(seed=seed)

        val_unlabeled = train_dataset.select(val_unlabeled_idxs)
        val_labeled = test_dataset.select(val_labeled_idxs)
        final_val_dataset = concatenate_datasets([val_unlabeled, val_labeled])
        final_val_dataset = final_val_dataset.shuffle(seed=seed)

        test_unlabeled = train_dataset.select(test_unlabeled_idxs)
        test_labeled = test_dataset.select(test_labeled_idxs)
        final_test_dataset = concatenate_datasets([test_unlabeled, test_labeled])
        final_test_dataset = final_test_dataset.shuffle(seed=seed)

        self.final_train_dataset = final_train_dataset
        self.final_test_dataset = final_test_dataset
        self.final_val_dataset = final_val_dataset

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, return_tensors="pt"
        )
        self.train_loader = DataLoader(
            final_train_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            final_val_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            final_test_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            drop_last=False,
        )
        self.positive_sampled_msg_ids = pd.read_csv(CARETAKER_PATH)
