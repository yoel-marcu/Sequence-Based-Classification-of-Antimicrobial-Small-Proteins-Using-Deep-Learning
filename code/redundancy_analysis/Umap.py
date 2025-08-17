import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
import os

HDF5_PATH = "/sci/labs/asafle/yoel.marcu2003/Project_G/embed_experiments/exp2/exp2_embeds.h5"
SAVE_PLOT = os.path.join(os.path.dirname(HDF5_PATH), "umap_projection_2d_slide.png")

def masked_mean(embedding):
    mask = (embedding.abs().sum(dim=-1) != 0).float()
    mask = mask.unsqueeze(-1)
    embedding = embedding * mask
    summed = embedding.sum(dim=0)
    count = mask.sum(dim=0).clamp(min=1e-6)
    return (summed / count).numpy()

def load_embeddings_and_labels(hdf5_path):
    vectors = []
    labels = []
    with h5py.File(hdf5_path, 'r') as f:
        for seq_id in f['embeddings'].keys():
            embedding = torch.tensor(f['embeddings'][seq_id][()], dtype=torch.float32)
            label = f['labels'][seq_id][()]
            vector = masked_mean(embedding)
            vectors.append(vector)
            labels.append(label)
    return np.array(vectors), np.array(labels)

def plot_umap_2d(X, y):
    # Sample 300 from each class
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.seed(42)
    pos_sample = np.random.choice(pos_idx, size=min(300, len(pos_idx)), replace=False)
    neg_sample = np.random.choice(neg_idx, size=min(300, len(neg_idx)), replace=False)
    sample_idx = np.concatenate([pos_sample, neg_sample])
    
    X_sampled = X[sample_idx]
    y_sampled = y[sample_idx]

    # UMAP to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_reduced = reducer.fit_transform(X_sampled)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(X_reduced[y_sampled == 0, 0], X_reduced[y_sampled == 0, 1],
                c='blue', alpha=0.6, label='Negative')
    plt.scatter(X_reduced[y_sampled == 1, 0], X_reduced[y_sampled == 1, 1],
                c='red', alpha=0.6, label='Positive')

    # Styling
    plt.title("2D UMAP of Protein Embeddings", fontsize=24, fontweight='bold')
    plt.xlabel("UMAP-1", fontsize=20)
    plt.ylabel("UMAP-2", fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=16)
    plt.gca().set_facecolor('white')
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()

    plt.savefig(SAVE_PLOT, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"? 2D UMAP plot saved to {SAVE_PLOT}")

if __name__ == "__main__":
    X, y = load_embeddings_and_labels(HDF5_PATH)
    plot_umap_2d(X, y)
