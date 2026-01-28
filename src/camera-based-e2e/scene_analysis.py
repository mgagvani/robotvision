import json
import numpy as np
import pickle
from sklearn.cluster import KMeans
from loader import WaymoE2E
import pandas as pd

def compute_features(scene_list):  # changed: now takes list of frames
    past_all = np.stack([scene["PAST"] for scene in scene_list], axis=0)  # tensor, (Frames, 16 timesteps, 6 values)
    past = past_all.reshape(-1, 6)  # reshapes into (F*16, 6)

    speed = np.linalg.norm(past[:, 2:4], axis=1)
    x_accel = past[:, 4]
    y_accel = past[:, 5]
    
    '''
    Curvature K = |dT/dS|, where T is unit tangent vector and S is the line
    T = v'(t)/|v'(t)| --> unit vector --> T = <costheta, sintheta>
    Any unit vector pointing in direction of theta has components <costheta, sintheta>
    (dT/dS) * (dtheta/dT) = (dtheta/dS)*<-sintheta, costheta>
    |<-sintheta, costheta>| = 1, thus K = dtheta/dS
    theta = arctan(y/x)
    theta(t) = arctan(dy/dt, dx/dt) --> direction car is MOVING
    dtheta/dt --> if this is larger, the turn is sharper
    dtheta/dS = (dtheta/dt) * (dt/dS)
    dS/dt = speed = sqrt((dx/dt)'^2+(dy/dt)^2)
    dt/dS = 1/speed
    Thus, curvature K = |dtheta/dt|/(speed) --> single scalar
    '''

    dx = np.gradient(past[:, 0])
    dy = np.gradient(past[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (((dx*dx + dy*dy)**1.5) + 1e-6)
    curvature = np.nan_to_num(curvature)
    curvature = curvature.mean()
    
    return {
        "mean_speed": speed.mean(),
        "speed_var": speed.var(),
        "x_accel": x_accel.mean(),
        "y_accel": y_accel.mean(),
        "curvature": curvature
    }


losses = json.load(open("scene_loss.json"))
idx = pickle.load(open("index_val.pkl", "rb"))

sorted_scenes = sorted(losses.items(), key=lambda x: x[1], reverse=True)
top_k_hardest = sorted_scenes
top_k_hardest_ids = [scene[0] for scene in top_k_hardest]

print(f"Hardest scenes: {top_k_hardest}")

val_dataset = WaymoE2E(
    batch_size=1,
    indexFile="index_val.pkl",
    data_dir="/scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0",
    images=False
)

scene_map = {}
for item in val_dataset:
    name = item["NAME"]
    if name not in scene_map:
        scene_map[name] = []   # changed: each scene maps to list of frames
    scene_map[name].append(item)  # changed: append frame
print(f"Mapped {len(scene_map)} scenes from dataset.")


all_features = []
for name in top_k_hardest_ids:
    if name not in scene_map:
        print(f"[WARNING] Scene {name} not found in dataset. Skipping.")
        continue
    scene_frames = scene_map[name]   # changed: use list of frames
    feat = compute_features(scene_frames)  # changed: compute on full sequence
    feat["loss"] = losses[name]
    feat["scene"] = name
    all_features.append(feat)

df = pd.DataFrame(all_features)
df.to_csv("scene_features.csv", index=False)
print("Saved all_scene_features.csv with:")
print(df.head())

# X = df[["mean_speed", "speed_var", "x_accel", "y_accel", "curvature"]].values
# kmeans = KMeans(n_clusters=5, random_state=42)
# cluster_labels = kmeans.fit_predict(X)
# df["cluster"] = cluster_labels
# df.to_csv("scene_features_clustered.csv", index=False)
# print("Saved scene_features_clustered.csv")

# cluster_summary = df.groupby("cluster")["loss"].mean().sort_values(ascending=False)
# print("\n=== Mean Loss per Cluster ===")
# print(cluster_summary)
# print("\nHardest cluster(s):")
# print(cluster_summary.index[:2].tolist())
