import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score
)
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import argparse
from CSVSequenceDataset import CSVSequenceDataset
from TrainingExperimentConfig import TrainingExperimentConfig
from typing import List
from sklearn.metrics import precision_score, recall_score, f1_score, auc
import matplotlib.pyplot as plt
import csv
from Networks import FixedMLP, FixedCombinedMLP, DynamicMLP, DynamicCombinedMLP
import json
import os
import argparse
import torch
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig



def mean_pool(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)

def median_pool(x: torch.Tensor) -> torch.Tensor:
    return x.median(dim=0).values

def max_pool(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=0).values

def min_pool(x: torch.Tensor) -> torch.Tensor:
    return x.min(dim=0).values

pooling_map = {
    "mean_pool": mean_pool,
    "median_pool": median_pool,
    "min_pool": min_pool,
    "max_pool": max_pool
}

def esmc_embedding(sequence, device):
    """Helper for ESMC model embeddings"""
    client = ESMC.from_pretrained("esmc_300m").to(device)
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings.squeeze()


def get_appropriate_network(row: pd.Series) -> nn.Module:
    if row["Fixed Network"]:
        if "+" in row["Pooling Functions"]:
            return FixedCombinedMLP(input_dim=960, pooling_fns=[pooling_map[row["Pooling Functions"].split("+")[0]],
                                           pooling_map[row["Pooling Functions"].split("+")[1]]])
        else:
            return FixedMLP(input_dim=960, pooling_fn=pooling_map[row["Pooling Functions"]])
    else:
        if "+" in row["Pooling Functions"]:
            return DynamicCombinedMLP(input_dim=960, pooling_fns=[pooling_map[row["Pooling Functions"].split("+")[0]],
                                           pooling_map[row["Pooling Functions"].split("+")[1]]])
        else:
            return DynamicMLP(input_dim=960, pooling_fn=pooling_map[row["Pooling Functions"]])
        

def parse_fasta_with_embeddings(filepath, device):
    """
    Returns:
        dict: { seq_id: {"sequence": str, "embedding": <torch.Tensor or np.ndarray>} }
    """
    sequences = {}
    current_id = None
    current_sequence = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # flush previous record
                if current_id is not None:
                    seq = "".join(current_sequence)
                    sequences[current_id] = {
                        "sequence": seq,
                        "embedding": esmc_embedding(seq, device)  # your function
                    }
                current_id = line[1:]  # drop '>'
                current_sequence = []
            else:
                current_sequence.append(line)

        # flush the last record
        if current_id is not None:
            seq = "".join(current_sequence)
            sequences[current_id] = {
                "sequence": seq,
                "embedding": esmc_embedding(seq, device)
            }

    return sequences


def get_local_path(old_path, old_root, new_root) -> str:
    old_path = os.path.normpath(old_path)
    if old_path.startswith(old_root):
        old_path = old_path.replace(old_root, new_root, 1)
    return old_path



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument('--sequences_fasta', type=str, required=True)
    parser.add_argument('--models_csv', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.models_csv)
    df = df[df["embedding_name"] == "esmc_300m"].reset_index(drop=True)

    sequences = parse_fasta_with_embeddings(args.sequences_fasta, device)
    seq_ids = sorted(sequences.keys())

    out_df = pd.DataFrame(index=[], columns=seq_ids, dtype=float)

    old_root = os.path.normpath("/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/new_models_results_fixed")
    new_root = os.path.normpath(r"C:\Users\yoelm\Project_G\new_models_results_fixed")

    # Helper to name models in the output table
    def model_row_name(row: pd.Series, ckpt_path: str) -> str:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        fixed = "Fixed" if str(row["Fixed Network"]).upper() == "TRUE" else "Dynamic"
        pool = row["Pooling Functions"]
        return f"{fixed}|{pool}|{base}"

    # Iterate models (no batching) and fill the row per model
    for _, row in df.iterrows():
        # Build model and load weights
        model = get_appropriate_network(row).to(device)
        ckpt_path = get_local_path(row["Model Path"], old_root, new_root)
        print(f"this is the path for {row['Model Name']}: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # Predict each sequence (one by one)
        preds = {}
        with torch.no_grad():
            for sid in seq_ids:
                emb = sequences[sid]["embedding"]
                # ensure tensor on the same device as the model
                x = torch.as_tensor(emb, dtype=torch.float32, device=device)
                out = model(x)                 # shape (1,) or scalar-like after .view(-1) inside
                if out.ndim == 0:
                    logit = out.item()
                else:
                    # (1,) -> float
                    logit = float(out.view(-1)[0].item())
                prob = float(torch.sigmoid(torch.tensor(logit)).item())
                preds[sid] = prob

        # Add row into the DataFrame
        out_df.loc[model_row_name(row, ckpt_path)] = [preds[s] for s in seq_ids]

    # Save results
    os.makedirs(args.root, exist_ok=True)
    out_csv = os.path.join(args.root, "predictions_prob_from_shani_second.csv")
    out_df.to_csv(out_csv, index=True)
    print(f"Saved probabilities table to: {out_csv}")






if __name__ == "__main__":
    main()