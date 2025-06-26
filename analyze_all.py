# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

import seaborn as sns
import numpy as np
from scipy.ndimage import generic_filter


# === CONFIGURATION ===
root_dir = "/extra/sscala/test_log/LGC"
FILTERS = {}  # e.g., {"dropout": [0.0]}
TOP_N = 15

# Control which curves to plot
PLOT_TRAIN_CURVES = True
PLOT_TEST_CURVES = True
PLOT_VALID_CURVES = True

sns.set(style="whitegrid")
all_runs = []

# === LOAD DATA ===
CACHE_FILE = "parsed_logs.pkl"
RELOAD = False  # Set to True if you want to force reloading logs

if os.path.exists(CACHE_FILE) and not RELOAD:
    df = pd.read_pickle(CACHE_FILE)
    print("Loaded cached DataFrame from disk.")
else:
    all_runs = []
    print("Loading DataFrame from path.")

    # === LOAD DATA ===
    for dirpath, _, filenames in os.walk(root_dir):
        log_data = {}
        log_file = next((f for f in filenames if f.endswith(".log")), None)
        test_file = next((f for f in filenames if "test" in f), None)
        train_file = next((f for f in filenames if "train" in f), None)
        valid_file = next((f for f in filenames if "valid" in f), None)

        if log_file:
            with open(os.path.join(dirpath, log_file)) as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        log_data[k.strip()] = v.strip()

        for ftype, key in [(test_file, "test_curve"), (train_file, "train_curve"), (valid_file, "valid_curve")]:
            if ftype:
                accs = []
                with open(os.path.join(dirpath, ftype)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[0].isdigit():
                            try:
                                accs.append(float(parts[2]))
                            except ValueError:
                                continue
                if accs:
                    log_data[key] = accs
                    if key == "test_curve":
                        log_data["final_test_acc"] = accs[-1]

        log_data["path"] = dirpath
        all_runs.append(log_data)

    # Convert to DataFrame
    df = pd.DataFrame(all_runs)
    for col in ["learning_rate", "dropout", "weight_decay", "k", "p", "final_test_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dataset_name", "p", "k", "final_test_acc"])
    df.to_pickle(CACHE_FILE)
    print("Parsed and cached DataFrame.")





# Apply optional filters
for key, values in FILTERS.items():
    if key in df.columns:
        df = df[df[key].isin(values)]

# === CONFIGURATION ===



GROUP_BY_KEYS = ["dataset_name", "p","k"]  # <-- Change this to average over chosen hyperparameters
#, "dropout", "learning_rate", "weight_decay"

## "dataset_name", "p", "k", "dropout", "learning_rate", "weight_decay", "final_test_acc", "path"


# === Compute average accuracy grouped by selected hyperparameters ===
avg_df = (
    df.dropna(subset=["final_test_acc"])
      .groupby(GROUP_BY_KEYS, as_index=False)["final_test_acc"]
      .agg( ["mean", "std"] )
      .reset_index()
      .rename(columns={"mean": "final_test_acc_mean", "std": "final_test_acc_std"})
)



# === Export top averaged configurations per dataset ===
top_configs = []
for dataset in avg_df["dataset_name"].unique():
    ddf = avg_df[avg_df["dataset_name"] == dataset]
    top = ddf.sort_values("final_test_acc_mean", ascending=False).head(TOP_N).copy()
    top_configs.append(top)

top_df = pd.concat(top_configs)

filename = f"top{TOP_N}_configs_groupedby_"+"_".join(GROUP_BY_KEYS)+".csv"

top_df.to_csv(filename, index=False)
print(f" Saved averaged top {TOP_N} configurations ( in {filename}")

# === Export top individual raw runs (no averaging) ===
top_full_configs = []
for dataset in df["dataset_name"].unique():
    ddf = df[df["dataset_name"] == dataset].dropna(subset=["final_test_acc"])
    top = ddf.sort_values("final_test_acc", ascending=False).head(TOP_N).copy()
    top_full_configs.append(top)

top_full_df = pd.concat(top_full_configs)

columns_to_keep = [
    "dataset_name", "p", "k", "dropout",
    "learning_rate", "weight_decay", "final_test_acc", "path"
]
top_full_df = top_full_df[columns_to_keep]
top_full_df.to_csv("top_full_configs.csv", index=False)
print(f"Saved top {TOP_N} raw configurations to top_full_configs.csv")



# === MAIN PLOTTING FUNCTION ===
def fill_nan_with_mean(matrix, mask=None):
    """Fill NaNs using the mean of their 8-connected neighbors."""
    array = matrix.values.copy()

    # Replace NaN with 0 and create a mask of where NaNs were
    nan_mask = np.isnan(array)

    # Compute a filtered version ignoring NaNs
    def nanmean_filter(values):
        vals = values[~np.isnan(values)]
        return np.mean(vals) if len(vals) > 0 else np.nan

    filled_array = array.copy()
    filled_array[nan_mask] = generic_filter(array, nanmean_filter, size=3, mode='constant', cval=np.NaN)[nan_mask]

    return pd.DataFrame(filled_array, index=matrix.index, columns=matrix.columns)


def compare_p_vs_hyperparam(df, hyperparam):
    if hyperparam not in df.columns:
        return

    df[hyperparam] = pd.to_numeric(df[hyperparam], errors='coerce')
    df = df.dropna(subset=[hyperparam, "p", "final_test_acc"])
    grouped_df = df.groupby(["dataset_name", "p", hyperparam], as_index=False).agg({"final_test_acc": "mean"})

    for dataset in grouped_df["dataset_name"].unique():
        ddf = grouped_df[grouped_df["dataset_name"] == dataset]

        # Lineplot
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=ddf, x="p", y="final_test_acc", hue=hyperparam, marker="o")
        plt.title(f"{dataset.upper()} - Accuracy vs p (Grouped by {hyperparam})")
        plt.xlabel("p")
        plt.ylabel("Final Test Accuracy")
        
        plt.legend(title=hyperparam)
        plt.tight_layout()
        plt.savefig(f"{dataset}_p_vs_acc_by_{hyperparam}.png")
        plt.close()

        # Heatmap
        pivot = ddf.pivot_table(index="p", columns=hyperparam, values="final_test_acc")
        if not pivot.empty:
            pivot_filled = fill_nan_with_mean(pivot)
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_filled, annot=True, fmt=".3f", cmap="YlOrRd")
            plt.title(f"{dataset.upper()} - Heatmap (p vs {hyperparam}) Final Test Accuracy")
            plt.xlabel(hyperparam)
            plt.ylabel("p")
            plt.tight_layout()
            plt.savefig(f"{dataset}_heatmap_p_vs_{hyperparam}.png")
            plt.close()

        # Error bars
        ddf_full = df[df["dataset_name"] == dataset]
        summary = ddf_full.groupby(["p", hyperparam])["final_test_acc"].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(10, 6))
        for h_val in sorted(summary[hyperparam].dropna().unique()):
            subset = summary[summary[hyperparam] == h_val]
            plt.errorbar(subset["p"], subset["mean"], yerr=subset["std"], label=f"{hyperparam}={h_val}", marker='o', capsize=4)

        plt.title(f"{dataset.upper()} - Accuracy vs p (Grouped by {hyperparam})")
        plt.xlabel("p")
        plt.ylabel("Test Accuracy (mean & std)")
        
        plt.legend(title=hyperparam)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dataset}_errorbar_p_vs_acc_by_{hyperparam}.png")
        plt.close()

# === APPLY TO HYPERPARAMETERS ===
for hyperparam in ["p", "k", "dropout", "learning_rate", "weight_decay"]:
    if hyperparam not in df.columns:
        continue

    df[hyperparam] = pd.to_numeric(df[hyperparam], errors='coerce')
    grouped_df = df.dropna(subset=["dataset_name", hyperparam, "final_test_acc"])

    for dataset in grouped_df["dataset_name"].unique():
        ddf = grouped_df[grouped_df["dataset_name"] == dataset]

        # === Group ONLY by the chosen hyperparam, averaging over all else ===
        summary = ddf.groupby(hyperparam)["final_test_acc"].agg(["mean", "std"]).reset_index()

        # === Plot with error bars ===
        plt.figure(figsize=(8, 5))
        plt.errorbar(summary[hyperparam], summary["mean"], yerr=summary["std"],
                     fmt='-o', capsize=4, label=f"{dataset}")
        plt.title(f"{dataset.upper()} - Accuracy vs {hyperparam} (Averaged over all others)")
        plt.xlabel(hyperparam)
        plt.ylabel("Test Accuracy (mean & std)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dataset}_mean_std_vs_{hyperparam}_global.png")
        plt.close()
print("global plots done")
# === PLOTS PER DATASET ===


for hp in ["k", "dropout", "learning_rate", "weight_decay"]:
    compare_p_vs_hyperparam(df, hp)

print("? All plots, summaries, and metrics saved successfully.")


def plot_avg_curves_by_p(df, dataset, curve_key, title_suffix, ylabel, filename_suffix):
    grouped = df.groupby("p")
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = TwoSlopeNorm(vmin=0.5, vcenter=2, vmax=3.5)
    cmap = plt.colormaps["coolwarm"]

    for p_val, group in grouped:
        curves = [row[curve_key] for _, row in group.iterrows() if isinstance(row.get(curve_key), list)]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        trimmed = [c[:min_len] for c in curves]
        avg_curve = [sum(x) / len(x) for x in zip(*trimmed)]
        color = cmap(norm(p_val))
        ax.plot(avg_curve, label=f"p={p_val}", color=color)

    ax.set_title(f"{dataset.upper()} - Avg {title_suffix} Accuracy by p")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, label="p value")
    cbar.set_ticks([0.5, 2, 3.5])
    cbar.set_ticklabels(["0", "2", "10"])

    plt.tight_layout()
    plt.savefig(f"{dataset}_{filename_suffix}.png")
    plt.close()

curve_configs = [
    ("train_curve", "Training", "Train Accuracy", "avg_train_curves_by_p", PLOT_TRAIN_CURVES),
    ("test_curve", "Test", "Test Accuracy", "avg_test_curves_by_p", PLOT_TEST_CURVES),
    ("valid_curve", "Validation", "Validation Accuracy", "avg_valid_curves_by_p", PLOT_VALID_CURVES),
]

for dataset in avg_df["dataset_name"].unique():
    ddf = df[df["dataset_name"] == dataset]

    for curve_key, title_suffix, ylabel, filename_suffix, do_plot in curve_configs:
        if do_plot:
            plot_avg_curves_by_p(
                ddf,
                dataset,
                curve_key=curve_key,
                title_suffix=title_suffix,
                ylabel=ylabel,
                filename_suffix=filename_suffix
            )

print(f"train({PLOT_TRAIN_CURVES}) test({PLOT_TEST_CURVES}) e val({PLOT_VALID_CURVES})  curves plotted.")

# === EXPORT REPRODUCIBLE COMMANDS ===
cmd_lines = []
for _, row in df.iterrows():
    dataset = row["dataset_name"]
    p = row["p"]
    k = row["k"]
    lr = row.get("learning_rate", "")
    wd = row.get("weight_decay", "")
    dropout = row.get("dropout", "")
    path = row["path"]

    cmd = (
        f"# Run for {dataset}, p={p}, k={k}\n"
        f"python LGC.py --dataset {dataset} --p {p} --k {k}"
    )
    if lr != "":
        cmd += f" --lr {lr}"
    if wd != "":
        cmd += f" --weight_decay {wd}"
    if dropout != "":
        cmd += f" --dropout {dropout}"

    cmd_lines.append({
        "dataset": dataset,
        "p": p,
        "k": k,
        "lr": lr,
        "dropout": dropout,
        "weight_decay": wd,
        "run_command": cmd,
        "log_dir": path
    })

pd.DataFrame(cmd_lines).to_csv("run_commands.csv", index=False)

print("? Exported reproducibility commands ? run_commands.csv")


