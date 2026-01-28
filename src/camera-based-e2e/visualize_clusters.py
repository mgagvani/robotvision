import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Load data
# -----------------------------
scene_feature_data = {}  # {scene_id: [mean_speed, speed_var, x_accel, y_accel, curvature, loss, cluster]}
cluster_dict = {}

with open("scene_features_clustered.csv") as f:
    for line in f:
        data = line.strip().split(",")
        if not data or "mean" in data[0]:
            continue

        scene_id = data[6]
        feats = [float(d) for d in data[0:6]]
        cluster = int(data[-1])

        scene_feature_data[scene_id] = feats + [cluster]

        if data[-1] in cluster_dict:
            cluster_dict[data[-1]].append(scene_id)
        else:
            cluster_dict[data[-1]] = [scene_id]

# -----------------------------
# Build X, y and remove all-zero rows
# -----------------------------
feature_names = ["mean_speed", "speed_var", "x_accel", "y_accel", "curvature", "loss"]

scene_ids_all = list(scene_feature_data.keys())
X_all = np.array([scene_feature_data[sid][:6] for sid in scene_ids_all], dtype=float)
y_all = np.array([scene_feature_data[sid][6] for sid in scene_ids_all], dtype=int)

# mask: keep rows where NOT all features are zero
# (use isclose to be safe with floating formatting)
nonzero_mask = ~np.all(np.isclose(X_all[:, :5], 0.0), axis=1)

scene_ids = [sid for sid, keep in zip(scene_ids_all, nonzero_mask) if keep]
X = X_all[nonzero_mask]
y = y_all[nonzero_mask]

print(f"Total scenes: {len(scene_ids_all)}")
print(f"Removed all-zero scenes: {np.sum(~nonzero_mask)}")
print(f"Remaining scenes: {len(scene_ids)}")

# Optional: if you also want to remove any NaNs/infs
finite_mask = np.isfinite(X).all(axis=1)
if not np.all(finite_mask):
    print(f"Removed non-finite rows: {np.sum(~finite_mask)}")
    X = X[finite_mask]
    y = y[finite_mask]
    scene_ids = [sid for sid, keep in zip(scene_ids, finite_mask) if keep]

# -----------------------------
# PCA + plots
# -----------------------------
# Create output folder
out_dir = "pca_outputs"
os.makedirs(out_dir, exist_ok=True)

# Standardize
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# PCA 2D
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xz)

# ---- Plot 1: PCA scatter (clusters) ----
plt.figure(figsize=(8, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=25)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA (2D) colored by cluster")
plt.colorbar(sc, label="cluster")
plt.tight_layout()

pca_scatter_path = os.path.join(out_dir, "pca_scatter_clusters.png")
plt.savefig(pca_scatter_path, dpi=200, bbox_inches="tight")
plt.close()

# ---- Quantify contributions ----
loadings = pca.components_.T  # (n_features, 2)
load_df = pd.DataFrame(loadings, index=feature_names, columns=["PC1_loading", "PC2_loading"])

load_df["PC1_contrib"] = load_df["PC1_loading"] ** 2
load_df["PC2_contrib"] = load_df["PC2_loading"] ** 2

var = pca.explained_variance_ratio_
load_df["PC1_var_weighted"] = load_df["PC1_contrib"] * var[0]
load_df["PC2_var_weighted"] = load_df["PC2_contrib"] * var[1]
load_df["PCA2D_importance_raw"] = load_df["PC1_var_weighted"] + load_df["PC2_var_weighted"]
load_df["PCA2D_importance"] = load_df["PCA2D_importance_raw"] / load_df["PCA2D_importance_raw"].sum()

print("\nExplained variance ratio:", var)
print("\nRanked feature importance for the 2D PCA view (variance-weighted):")
print(load_df.sort_values("PCA2D_importance", ascending=False)[
    ["PCA2D_importance", "PC1_contrib", "PC2_contrib", "PC1_loading", "PC2_loading"]
])

# Save the contributions table too (useful for writeups)
contrib_csv_path = os.path.join(out_dir, "pca_feature_contributions.csv")
load_df.sort_values("PCA2D_importance", ascending=False).to_csv(contrib_csv_path)

# ---- Plot 2: Bar chart of PCA2D importance ----
plt.figure(figsize=(8, 4))
load_df.sort_values("PCA2D_importance", ascending=False)["PCA2D_importance"].plot(kind="bar")
plt.ylabel("Variance-weighted contribution to PCA(PC1,PC2)")
plt.title("Which original variables drive the 2D PCA separation?")
plt.tight_layout()

importance_bar_path = os.path.join(out_dir, "pca2d_feature_importance_bar.png")
plt.savefig(importance_bar_path, dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved files:")
print(" -", pca_scatter_path)
print(" -", importance_bar_path)
print(" -", contrib_csv_path)
