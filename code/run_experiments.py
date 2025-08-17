
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


def mean_pool(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)

def median_pool(x: torch.Tensor) -> torch.Tensor:
    return x.median(dim=0).values

def max_pool(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=0).values

def min_pool(x: torch.Tensor) -> torch.Tensor:
    return x.min(dim=0).values


def get_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="ESM Model Training")
    parser.add_argument("--csv_path_train", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--csv_path_test", type=str, required=True, help="Path to testing CSV file")
    parser.add_argument("--root", type=str, required=True)
    return parser.parse_args()
    


def code_to_embedding(code_batch, embedding_path: str, label_batch=None) -> torch.Tensor:
    """
    Load embeddings (and optionally labels) for a batch of codes.
    Returns:
        X: Tensor of shape (B, D)
        y: Tensor of shape (B,) [if label_batch is provided]
    Skips entries with missing embedding files.
    """
    embeddings = []
    valid_labels = []

    for i, code in enumerate(code_batch):
        if isinstance(code, torch.Tensor):  # batch of tensors
            code = code.item() if code.ndim == 0 else code
        code = str(code)
        path = os.path.join(embedding_path, f"{code}.pt")
        if os.path.exists(path):
            emb = torch.load(path)
            if emb.ndim == 2:
                emb = emb.mean(dim=0)
            embeddings.append(emb)
            if label_batch is not None:
                valid_labels.append(label_batch[i])
        else:
            print(f"[Warning] Missing embedding for: {code}")

    if not embeddings:
        raise ValueError("No valid embeddings found in batch.")

    X = torch.stack(embeddings)
    if label_batch is not None:
        y = torch.stack(valid_labels) if isinstance(label_batch[0], torch.Tensor) else torch.tensor(valid_labels)
        return X, y
    return X


def folder_creator(experiment_name: str, root: str) -> str:
    """Create a folder for the experiment results."""
    folder_path = os.path.join(root, experiment_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def train_experiments(experiments: List[TrainingExperimentConfig], loader: DataLoader, device: torch.device, epochs: int = 20, lr: float = 1e-4, save: bool = False):
    """ Train all experiments on the provided DataLoader.
    Args:  
        experiments (List[TrainingExperimentConfig]): List of training configurations.
        loader (DataLoader): DataLoader for the training data.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
    """
    for exp in experiments:
        exp.model = exp.model_factory().to(device)
        exp.optimizer = torch.optim.Adam(exp.model.parameters(), lr=lr)
        exp.fold_labels = []
        exp.fold_preds = []
        exp.model.train()
    
    for _ in range(epochs): 
        emb_dir = experiments[0].embedding_path
        X, y_filtered = None, None
        for X_codes, y in loader:
            X, y_filtered = code_to_embedding(X_codes, emb_dir, y)
            X, y_filtered = X.to(device), y_filtered.to(device)
            for exp in experiments:
                if exp.embedding_path != emb_dir:
                    emb_dir = exp.embedding_path
                    X, y_filtered = code_to_embedding(X_codes, emb_dir, y)
                    X, y_filtered = X.to(device), y_filtered.to(device)
                exp.optimizer.zero_grad()
                loss = exp.criterion(exp.model(X), y_filtered)
                loss.backward()
                exp.optimizer.step()
    if save:
        for exp in experiments:
            model_path = os.path.join(folder_creator(exp.name, exp.root), f"{exp.name}_final_model.pth")
            torch.save(exp.model.state_dict(), model_path)
            print(f"Saved model for {exp.name} at {model_path}")


def test_experiments(experiments: List[TrainingExperimentConfig], loader: DataLoader, device: torch.device, fold_index: int = None):
    """ Test all experiments on the provided DataLoader.
    Args:  
        experiments (List[TrainingExperimentConfig]): List of training configurations.
        loader (DataLoader): DataLoader for the test data.
        device (torch.device): Device to run the testing on (CPU or GPU).
    """
    for exp in experiments:
        exp.model.eval()
        exp.fold_labels = []
        exp.fold_preds = []
    with torch.no_grad():
        emb_dir = experiments[0].embedding_path
        X, y_filtered = None, None
        for X_codes, y in loader:
            X, y_filtered = code_to_embedding(X_codes, emb_dir, y)
            X, y_filtered = X.to(device), y_filtered.to(device)
            for exp in experiments:
                if exp.embedding_path != emb_dir:
                    emb_dir = exp.embedding_path
                    X, y_filtered = code_to_embedding(X_codes, emb_dir, y)
                    X, y_filtered = X.to(device), y_filtered.to(device)
                prediction = torch.sigmoid(exp.model(X)).cpu().numpy()
                exp.fold_preds.extend(prediction)
                exp.fold_labels.extend(y_filtered.cpu().numpy())
    
    if fold_index is not None:
        for exp in experiments:
            fpr, tpr, _ = roc_curve(exp.fold_labels, exp.fold_preds)
            y_true = np.array(exp.fold_labels)
            y_pred = np.array(exp.fold_preds) > 0.5
            exp.k_fold_predictions[f"fold {fold_index}"] = {
                "probabilities": exp.fold_preds,
                "labels": exp.fold_labels,
                "fpr": fpr,
                "tpr": tpr, 
                "auc": roc_auc_score(exp.fold_labels, exp.fold_preds),
                "confusion_matrix": confusion_matrix(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "accuracy": accuracy_score(y_true, y_pred)
            }
    else:
        for exp in experiments:
            all_labels = np.array(exp.fold_labels)
            all_probs = np.array(exp.fold_preds)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            y_true = all_labels
            y_pred = all_probs > 0.5
            exp.final_results = {
                "probabilities": all_probs,
                "labels": all_labels,
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(all_labels, all_probs),
                "confusion_matrix": confusion_matrix(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "accuracy": accuracy_score(y_true, y_pred)
            }


def main():
    # === Parameters ===
    
    args = get_arguments()
    CSV_PATH_TRAIN = args.csv_path_train
    CSV_PATH_TEST = args.csv_path_test
    ROOT = args.root
    EPOCHS = 20
    LR = 1e-4
    POS = 996
    NEG = 2558
    experiments: List[TrainingExperimentConfig] = [
    ]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    POS_WEIGHT = torch.tensor([NEG / POS], device=DEVICE)


    experiments: List[TrainingExperimentConfig] = []

    # Embedding sources
    embedding_sources = {
        "esm2_t6":     ("/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/esm2_t6_8M_UR50D/esm_embeddings", 320),
        "esm2_t30":    ("/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/esm2_t30_150M_UR50D/esm_embeddings", 640),
        "esm2_t36":    ("/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/esm2_t36_3B_UR50D/esm_embeddings", 2560),
        "esmc_300m":   ("/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/ESMC_300m/esmc_embeddings", 960),
        "esmc_600m":   ("/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/ESMC_600m/esmc_embeddings", 1152)
    }

    # Pooling functions
    pooling_options = {
        "mean": mean_pool,
        "median": median_pool,
        "min": min_pool,
        "max": max_pool
    }

    pooling_pair_options = [
        ("mean", "median"),
        ("mean", "min"),
        ("mean", "max"),
        ("median", "min"),
        ("median", "max"),
        ("min", "max")
    ]

    # Create experiments for each embedding source and neural network configuration
    for emb_name, (emb_path, input_dim) in embedding_sources.items():
        # === Combined (mean + median) ===
        for weighted in [False, True]:
            loss_name = "weighted" if weighted else "unweighted"
            criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT) if weighted else nn.BCEWithLogitsLoss()

            # One pooling networks:
            for pooling_name, pooling_fn in pooling_options.items():

                # Create a fixed network with one pooling function
                experiments.append(TrainingExperimentConfig(
                    name=f"{emb_name}_fixed_{pooling_name}_{loss_name}",
                    embedding_name=emb_name,
                    root=ROOT,
                    pooling_fns=[pooling_fn],
                    network_type="fixed",
                    weighted=weighted,
                    fixed_network=True,
                    embedding_path=emb_path,
                    model_factory=lambda dim=input_dim, pf=pooling_fn: FixedMLP(input_dim=dim, pooling_fn=pf),
                    criterion=criterion
                ))

                # Create a dynamic network with one pooling function
                experiments.append(TrainingExperimentConfig(
                    name=f"{emb_name}_dynamic_{pooling_name}_{loss_name}",
                    embedding_name=emb_name,
                    root=ROOT,
                    pooling_fns=[pooling_fn],
                    network_type="dynamic",
                    weighted=weighted,
                    fixed_network=False,
                    embedding_path=emb_path,
                    model_factory=lambda dim=input_dim, pf=pooling_fn: DynamicMLP(input_dim=dim, pooling_fn=pf),
                    criterion=criterion
                ))

            # Combined pooling networks:
            for pooling_name1, pooling_name2 in pooling_pair_options:
                pooling_fn1 = pooling_options[pooling_name1]
                pooling_fn2 = pooling_options[pooling_name2]

                # Create a fixed network with two pooling functions
                experiments.append(TrainingExperimentConfig(
                    name=f"{emb_name}_fixed_{pooling_name1}-{pooling_name2}_{loss_name}",
                    embedding_name=emb_name,
                    root=ROOT,
                    pooling_fns=[pooling_fn1, pooling_fn2],
                    network_type="fixed",
                    weighted=weighted,
                    fixed_network=True,
                    embedding_path=emb_path,
                    model_factory=lambda dim=input_dim, pf1=pooling_fn1, pf2=pooling_fn2:
                        FixedCombinedMLP(input_dim=dim, pooling_fns=[pf1, pf2]),
                    criterion=criterion
                ))

                # Create a dynamic network with two pooling functions
                experiments.append(TrainingExperimentConfig(
                    name=f"{emb_name}_dynamic_{pooling_name1}-{pooling_name2}_{loss_name}",
                    embedding_name=emb_name,
                    root=ROOT,
                    pooling_fns=[pooling_fn1, pooling_fn2],
                    network_type="dynamic",
                    weighted=weighted,
                    fixed_network=False,
                    embedding_path=emb_path,
                    model_factory=lambda dim=input_dim, pf1=pooling_fn1, pf2=pooling_fn2:
                        DynamicCombinedMLP(input_dim=dim, pooling_fns=[pf1, pf2]),
                    criterion=criterion
                ))

    experiments.sort() # Sort by embedding path, thus we will be able to do the least amount of reloading during loops

    # Load datasets and prepare for k-fold cross-validation
    df_train = pd.read_csv(CSV_PATH_TRAIN).dropna(subset=["SEQUENCE", "LABEL", "ID"])
    df_test = pd.read_csv(CSV_PATH_TEST).dropna(subset=["SEQUENCE", "LABEL", "ID"])
    train_dataset = CSVSequenceDataset(df_train)
    test_dataset = CSVSequenceDataset(df_test)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # k-fold croos validation:
    for fold_idx, (fold_train, fold_val) in enumerate(kfold.split(train_dataset)):

        # Prepare subsets for this fold
        train_subset = Subset(train_dataset, fold_train)
        validation_subset = Subset(train_dataset, fold_val)
        fold_train_loader = DataLoader(train_subset)
        fold_validation_loader = DataLoader(validation_subset)
        
        # Training loop for this fold
        train_experiments(experiments, fold_train_loader, DEVICE, EPOCHS, LR)
            
        # Test model on this fold's validation set
        test_experiments(experiments, fold_validation_loader, DEVICE, fold_index=fold_idx)
    

    print("\n=== Training each model on full train set and evaluating on test set ===")
    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    # === Train each experiment on the full training set ===
    train_experiments(experiments, train_loader, DEVICE, EPOCHS, LR, save=True)

    # === Evaluation loop (on test set)
    test_experiments(experiments, test_loader, DEVICE)

    # === Save final model and results for each experiment ===
    for exp in experiments:
        with open(os.path.join(folder_creator(exp.name, ROOT), f"{exp.name}_experiment_summary.pkl"), "wb") as f:
            pickle.dump({
                "name": exp.name,
                "final_results": exp.final_results,
                "k_fold_predictions": exp.k_fold_predictions
            }, f)

    # === Summarize all results into a comparison CSV ===
    summary_rows = []

    for exp in experiments:
        final_report_path = os.path.join(folder_creator(exp.name, ROOT), f"{exp.name}_final_model.pth")

        row = {
            "Model Name": exp.name,
            "embedding_name": exp.embedding_name,
            "Pooling Functions": "+".join([fn.__name__ for fn in exp.pooling_fns]),
            "Network Type": exp.network_type,
            "Weighted": exp.weighted,
            "Fixed Network": exp.fixed_network,
            "Final AUC": exp.final_results["auc"],
            "Final Accuracy": exp.final_results["accuracy"],
            "Final Precision": exp.final_results["precision"],
            "Final Recall": exp.final_results["recall"],
            "Final F1": exp.final_results["f1"],
            "Mean CV F1": np.mean([v["f1"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "f1" in v]),
            "Std CV F1": np.std([v["f1"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "f1" in v]),
            "Mean CV Precision": np.mean([v["precision"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "precision" in v]),
            "Std CV Precision": np.std([v["precision"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "precision" in v]),
            "Mean CV Recall": np.mean([v["recall"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "recall" in v]),
            "Std CV Recall": np.std([v["recall"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "recall" in v]),
            "Mean CV AUC": np.mean([v["auc"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "auc" in v]),
            "Std CV AUC": np.std([v["auc"] for fold, v in exp.k_fold_predictions.items() if isinstance(v, dict) and "auc" in v]),
            "Model Path": final_report_path
        }

        summary_rows.append(row)

        summary_path = os.path.join(ROOT, "new_experiments_results_summary.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSummary CSV saved to: {summary_path}")






if __name__ == "__main__":
    main()

