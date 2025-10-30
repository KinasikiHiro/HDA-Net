# visualize_metrics.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import pi

# ----------------------
# Paths
# ----------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "joint_evaluate")
CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# Load metrics.csv
# ----------------------
models_data = {"baseline": [], "joint": []}
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row["model"]
        metrics = {k: float(v) for k, v in row.items() if k != "model" and k != "case"}
        models_data[model].append(metrics)

# ----------------------
# Prepare metric names
# ----------------------
metrics_list = list(models_data["baseline"][0].keys())
plot_metrics = [m for m in metrics_list if "cls1" in m or "cls2" in m]

# Mapping cls1/cls2 -> Liver/Tumor
metrics_named = []
for m in plot_metrics:
    if "cls1" in m:
        metrics_named.append(m.replace("cls1","Liver"))
    elif "cls2" in m:
        metrics_named.append(m.replace("cls2","Tumor"))

# ----------------------
# Boxplot (per-case distribution)
# ----------------------
plt.figure(figsize=(12,6))
n_metrics = len(plot_metrics)
positions_base = np.arange(n_metrics) * 2.0
positions_joint = positions_base + 0.8

data_base = [[r[m] for r in models_data["baseline"]] for m in plot_metrics]
data_joint = [[r[m] for r in models_data["joint"]] for m in plot_metrics]

bp_base = plt.boxplot(data_base, positions=positions_base, widths=0.6, patch_artist=True)
bp_joint = plt.boxplot(data_joint, positions=positions_joint, widths=0.6, patch_artist=True)

# 上色
for box in bp_base['boxes']:
    box.set(facecolor='blue', alpha=0.5)
for box in bp_joint['boxes']:
    box.set(facecolor='orange', alpha=0.5)

plt.xticks(positions_base + 0.4, metrics_named, rotation=30)
plt.ylabel("Metric value")
plt.title("Model Comparison per-case")

# 手动图例
legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Baseline'),
                   Line2D([0], [0], color='orange', lw=4, label='Joint')]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics_boxplot.png"), dpi=300)
plt.close()

# ----------------------
# Radar chart (average metrics)
# ----------------------
def compute_avg(model_list):
    avg = {}
    for m in plot_metrics:
        avg[m] = np.mean([r[m] for r in model_list])
    return avg

avg_base = compute_avg(models_data["baseline"])
avg_joint = compute_avg(models_data["joint"])

# 准备雷达图数据
labels = metrics_named
n = len(labels)
angles = np.linspace(0, 2*pi, n, endpoint=False).tolist()
angles += angles[:1]  # 闭合

base_vals = [avg_base[m.replace("Liver","cls1").replace("Tumor","cls2")] for m in labels]
base_vals += base_vals[:1]
joint_vals = [avg_joint[m.replace("Liver","cls1").replace("Tumor","cls2")] for m in labels]
joint_vals += joint_vals[:1]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

ax.plot(angles, base_vals, label="Baseline", color="blue", linewidth=2)
ax.fill(angles, base_vals, alpha=0.25, color="blue")
ax.plot(angles, joint_vals, label="Joint", color="orange", linewidth=2)
ax.fill(angles, joint_vals, alpha=0.25, color="orange")

# 设置每个角度的标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1)
plt.title("Average Metric Comparison (Liver / Tumor)", fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics_radar.png"), dpi=300)
plt.close()

print(f"[INFO] Figures saved → {OUT_DIR}")
