import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Define the 20 standard amino acids
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Load the dataset
df = pd.read_csv("/sci/labs/asafle/yoel.marcu2003/Project_G/procecced_source/chosen_systems.csv", index_col=False)

# Ensure expected columns exist
assert "sequence" in df.columns and "label" in df.columns, "Missing required columns"

# Helper: Count AAs in a list of sequences
def count_amino_acids(sequences):
    counter = Counter()
    for seq in sequences:
        if isinstance(seq, str):
            filtered_seq = [aa for aa in seq if aa in STANDARD_AAS]
            counter.update(filtered_seq)
    total = sum(counter.values())
    return counter, total

# Split by label
pos_seqs = df[df["label"] == 1]["sequence"]
neg_seqs = df[df["label"] == 0]["sequence"]

# Count AAs
pos_counts, pos_total = count_amino_acids(pos_seqs)
neg_counts, neg_total = count_amino_acids(neg_seqs)

# Prepare result DataFrame
aa_data = []
for aa in STANDARD_AAS:
    pos_count = pos_counts.get(aa, 0)
    neg_count = neg_counts.get(aa, 0)
    aa_data.append({
        "AA": aa,
        "Positive Count": pos_count,
        "Positive %": 100 * pos_count / pos_total,
        "Negative Count": neg_count,
        "Negative %": 100 * neg_count / neg_total
    })

result_df = pd.DataFrame(aa_data)


# Plot
plt.figure(figsize=(12, 6))
bar_width = 0.4
index = range(len(STANDARD_AAS))

# Plot counts
plt.bar(index, [pos_counts.get(aa, 0) for aa in STANDARD_AAS], bar_width, label='Positive', alpha=0.7)
plt.bar([i + bar_width for i in index], [neg_counts.get(aa, 0) for aa in STANDARD_AAS], bar_width, label='Negative', alpha=0.7)
plt.xlabel("Amino Acid")
plt.ylabel("Count")
plt.title("Amino Acid Count Distribution")
plt.xticks([i + bar_width / 2 for i in index], STANDARD_AAS)
plt.legend()
plt.tight_layout()
plt.savefig("/sci/labs/asafle/yoel.marcu2003/Project_G/procecced_source/AA_dist_count.png")
plt.show()

# Optional: plot percentages
plt.figure(figsize=(12, 6))
plt.bar(index, [100 * pos_counts.get(aa, 0) / pos_total for aa in STANDARD_AAS], bar_width, label='Positive %', alpha=0.7)
plt.bar([i + bar_width for i in index], [100 * neg_counts.get(aa, 0) / neg_total for aa in STANDARD_AAS], bar_width, label='Negative %', alpha=0.7)
plt.xlabel("Amino Acid")
plt.ylabel("Percentage")
plt.title("Amino Acid Frequency (%)")
plt.xticks([i + bar_width / 2 for i in index], STANDARD_AAS)
plt.legend()
plt.tight_layout()
plt.savefig("/sci/labs/asafle/yoel.marcu2003/Project_G/procecced_source/AA_dist_perc.png")
plt.show()
