import os
import pickle
from pathlib import Path

import numpy as np
import shap
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pu_data_split import Positive_Unlabeled_Data_Split

SCRIPT_DIR = Path(__file__).parent.resolve()
dotenv_path = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
BATCH_SIZE = 2048
SHAP_BATCH_SIZE = 2048
MAX_LEN = 512

if not all([MODEL_NAME, OUTPUT_PATH, EXPERIMENT_NAME]):
    missing = [
        var
        for var, val in {
            "MODEL_NAME": MODEL_NAME,
            "OUTPUT_PATH": OUTPUT_PATH,
            "EXPERIMENT_NAME": EXPERIMENT_NAME,
        }.items()
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def run_shap_on_gpu(
    gpu_id,
    texts,
    output_dir,
    model_name,
    best_path,
    max_len,
    batch_size,
    shap_batch_size,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    def predict(input_text):
        if isinstance(input_text, np.ndarray):
            input_text = input_text.tolist()
        inputs = tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(), autocast(device_type=device.type):
            outputs = model(**inputs).logits
            probs = torch.softmax(outputs, dim=-1)
        return probs.cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict, masker)

    for i in tqdm(range(0, len(texts), batch_size), desc=f"GPU {gpu_id}"):
        batch_texts = texts[i : i + batch_size]
        shap_values_batch = explainer(batch_texts, batch_size=shap_batch_size)
        with open(f"{output_dir}/shap_gpu{gpu_id}_batch_{i}.pkl", "wb") as f:
            pickle.dump(shap_values_batch, f)


if __name__ == "__main__":
    all_data = Positive_Unlabeled_Data_Split(
        model_name=MODEL_NAME, batch_size=BATCH_SIZE
    )

    output_dir = f"{OUTPUT_PATH}/output_{MODEL_NAME}_{EXPERIMENT_NAME}"

    num_gpus = torch.cuda.device_count()
    best_path = os.path.join(output_dir, "best_prevalence_model.pth")

    # Split test data across GPUs
    print(len(all_data.final_test_dataset["content"]))
    test = all_data.final_test_dataset["content"]
    splits = np.array_split(test, num_gpus)

    # Spawn one process per GPU
    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=run_shap_on_gpu,
            args=(
                gpu_id,
                splits[gpu_id],
                output_dir,
                MODEL_NAME,
                best_path,
                MAX_LEN,
                BATCH_SIZE,
                SHAP_BATCH_SIZE,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
