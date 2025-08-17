#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import Counter
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------
# Greedy MIS utilities
# ---------------------------

def greedy_mis_from_order(G, order):
    """Return a (maximal) independent set using a greedy scan over `order`."""
    selected = set()
    blocked = set()
    add = selected.add
    block = blocked.add
    neigh = G.neighbors
    for u in order:
        if u not in blocked:
            add(u)
            block(u)
            for v in neigh(u):
                block(v)
    return selected

def greedy_mis_random(G, nodes, rng):
    """Randomized greedy MIS: shuffle node order, then greedy select."""
    order = nodes[:]
    rng.shuffle(order)
    return greedy_mis_from_order(G, order)

def greedy_mis_degree_ascending(G):
    """Greedy MIS with nodes processed by ascending degree (common heuristic)."""
    order = sorted(G.nodes(), key=lambda u: G.degree(u))
    return greedy_mis_from_order(G, order)

def greedy_mis_degree_descending(G):
    """Greedy MIS with nodes processed by descending degree."""
    order = sorted(G.nodes(), key=lambda u: G.degree(u), reverse=True)
    return greedy_mis_from_order(G, order)

# Worker for multiprocessing
def _worker_random_trial(args):
    G, nodes, seed = args
    rng = random.Random(seed)
    return greedy_mis_random(G, nodes, rng)

# ---------------------------
# Graph metadata
# ---------------------------

def graph_metadata(G):
    N = G.number_of_nodes()
    M = G.number_of_edges()
    density = nx.density(G) if N > 1 else 0.0
    avg_deg = (2.0 * M / N) if N > 0 else 0.0

    # Connected components
    comps = [len(c) for c in nx.connected_components(G)]
    comps.sort(reverse=True)
    n_comp = len(comps)
    largest = comps[0] if comps else 0
    comp_stats = {
        "num_components": n_comp,
        "largest_component_size": largest,
        "component_size_mean": float(np.mean(comps)) if comps else 0.0,
        "component_size_median": float(np.median(comps)) if comps else 0.0,
        "component_size_p95": float(np.percentile(comps, 95)) if comps else 0.0,
    }

    # Degree distribution stats
    degs = [d for _, d in G.degree()]
    deg_stats = {
        "avg_degree": float(avg_deg),
        "min_degree": int(min(degs)) if degs else 0,
        "max_degree": int(max(degs)) if degs else 0,
        "degree_p50": float(np.percentile(degs, 50)) if degs else 0.0,
        "degree_p90": float(np.percentile(degs, 90)) if degs else 0.0,
        "degree_p99": float(np.percentile(degs, 99)) if degs else 0.0,
    }

    return {
        "num_nodes": N,
        "num_edges": M,
        "density": float(density),
        **comp_stats,
        **deg_stats,
    }

# ---------------------------
# Main routine
# ---------------------------

def run_mis_analysis(
    tsv_path,
    out_ids_path,
    out_report_json,
    trials=10000,
    workers=None,
    seed=42,
    drop_self_hits=True,
    undirected=True,
):
    # Load BLAST tabular (outfmt 6)
    cols = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=cols)

    # Optionally drop self-hits (qseqid == sseqid)
    if drop_self_hits:
        df = df[df["qseqid"] != df["sseqid"]]

    # Build undirected graph over unique pairs
    # Deduplicate edges by unordered pair
    if undirected:
        # Make a canonical edge key (min, max)
        pairs = pd.DataFrame({
            "u": np.where(df["qseqid"] < df["sseqid"], df["qseqid"], df["sseqid"]),
            "v": np.where(df["qseqid"] < df["sseqid"], df["sseqid"], df["qseqid"]),
        }).drop_duplicates()
        edges = list(map(tuple, pairs.to_numpy()))
        G = nx.Graph()
        G.add_edges_from(edges)
    else:
        edges = list(zip(df["qseqid"], df["sseqid"]))
        G = nx.DiGraph()
        G.add_edges_from(edges)
        # Convert to undirected for MIS anyway
        G = G.to_undirected()

    # Ensure we include isolated nodes: gather all IDs present in q or s
    all_ids = set(df["qseqid"]).union(set(df["sseqid"]))
    G.add_nodes_from(all_ids)

    # Graph metadata
    meta = graph_metadata(G)

    # Prepare for trials
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        raise RuntimeError("Empty graph; check your input file/filters.")

    # Deterministic degree-based MIS
    mis_deg_asc = greedy_mis_degree_ascending(G)
    mis_deg_desc = greedy_mis_degree_descending(G)

    # Randomized trials (parallel)
    if workers is None:
        workers = min(cpu_count(), 16)

    rng = random.Random(seed)
    trial_args = []
    for t in range(trials):
        trial_seed = rng.randrange(1 << 30)
        trial_args.append((G, nodes, trial_seed))

    best_set = set()
    sizes = []

    if trials > 0:
        with Pool(processes=workers) as pool:
            for mis_set in pool.imap_unordered(_worker_random_trial, trial_args, chunksize=64):
                sizes.append(len(mis_set))
                if len(mis_set) > len(best_set):
                    best_set = mis_set

    # Stats over trials
    if sizes:
        sizes_arr = np.array(sizes, dtype=np.int64)
        mis_stats = {
            "trials": trials,
            "size_min": int(sizes_arr.min()),
            "size_mean": float(sizes_arr.mean()),
            "size_std": float(sizes_arr.std(ddof=1)) if len(sizes_arr) > 1 else 0.0,
            "size_max": int(sizes_arr.max()),
        }
    else:
        mis_stats = {
            "trials": 0,
            "size_min": 0,
            "size_mean": 0.0,
            "size_std": 0.0,
            "size_max": 0,
        }

    # Compare with deterministic heuristics
    deg_asc_size = len(mis_deg_asc)
    deg_desc_size = len(mis_deg_desc)
    heuristic_best_set = mis_deg_asc if deg_asc_size >= deg_desc_size else mis_deg_desc
    heuristic_best_name = "degree_ascending" if deg_asc_size >= deg_desc_size else "degree_descending"
    heuristic_best_size = max(deg_asc_size, deg_desc_size)

    # Choose final MIS to export (best of trials vs best heuristic)
    if len(best_set) >= heuristic_best_size:
        final_ids = best_set
        final_source = "randomized_trials"
        final_size = len(best_set)
    else:
        final_ids = heuristic_best_set
        final_source = heuristic_best_name
        final_size = heuristic_best_size

    # Save selected IDs
    with open(out_ids_path, "w") as f:
        for sid in sorted(final_ids):
            f.write(f"{sid}\n")

    # Build report
    report = {
        "input_tsv": os.path.abspath(tsv_path),
        "num_sequences_in_edges": int(len(all_ids)),
        "graph": meta,
        "mis_random_trials": mis_stats,
        "mis_degree_ascending_size": int(deg_asc_size),
        "mis_degree_descending_size": int(deg_desc_size),
        "chosen_final": {
            "source": final_source,
            "size": int(final_size),
            "ids_path": os.path.abspath(out_ids_path),
        },
    }

    # Write JSON report
    with open(out_report_json, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print("\n=== Graph summary ===")
    for k, v in report["graph"].items():
        print(f"{k:28s}: {v}")
    print("\n=== MIS (random trials) ===")
    for k, v in report["mis_random_trials"].items():
        print(f"{k:28s}: {v}")
    print("\n=== Deterministic heuristics ===")
    print(f"{'degree_ascending_size':28s}: {deg_asc_size}")
    print(f"{'degree_descending_size':28s}: {deg_desc_size}")
    print("\n=== Final selection ===")
    print(f"source: {final_source} | size: {final_size}")
    print(f"IDs written to: {os.path.abspath(out_ids_path)}")
    print(f"Report JSON : {os.path.abspath(out_report_json)}")


def parse_args():
    ap = argparse.ArgumentParser(description="MIS analysis over BLAST similarity graph")
    ap.add_argument("--tsv", required=True, help="BLAST outfmt 6 file (TSV)")
    ap.add_argument("--out_ids", required=True, help="Path to write final MIS IDs (one per line)")
    ap.add_argument("--out_report", required=True, help="Path to write JSON report")
    ap.add_argument("--trials", type=int, default=10000, help="Number of randomized greedy trials")
    ap.add_argument("--workers", type=int, default=None, help="Parallel workers (default=min(16, CPU count))")
    ap.add_argument("--seed", type=int, default=42, help="Seed for trial seeds generator")
    ap.add_argument("--keep_self_hits", action="store_true", help="Keep self-hits (off by default)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mis_analysis(
        tsv_path=args.tsv,
        out_ids_path=args.out_ids,
        out_report_json=args.out_report,
        trials=args.trials,
        workers=args.workers,
        seed=args.seed,
        drop_self_hits=(not args.keep_self_hits),
        undirected=True,
    )
