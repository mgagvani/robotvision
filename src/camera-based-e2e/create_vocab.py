import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

from loader import WaymoE2E
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

def get_all_trajectories():
    # Instantiate dataloader w/ n_items None to get everything
    dataset = WaymoE2E(indexFile='index_train.pkl', 
                       data_dir='/anvil/scratch/x-mgagvani/wod/waymo_end_to_end_camera_v1_0_0/waymo_open_dataset_end_to_end_camera_v_1_0_0', 
                       images=False, 
                       n_items=None # set to None eventually
                       )

    dataloader = DataLoader(dataset, batch_size=512, num_workers=16, persistent_workers=True, pin_memory=True)

    # iterate over dataloader, only obtain GT future (x, y)
    all_traj = []
    for batch in tqdm(dataloader, desc="Collecting trajectories...", total=len(dataloader)):
        future = batch['FUTURE']  # (B, 20, 2)
        all_traj.append(future.cpu().numpy())
    all_traj = np.concatenate(all_traj, axis=0)  # (N, 20, 2)
    return all_traj

def plot_trajectories(trajectories, name="trajectories_samples.png"):
    print("Plotting trajectories...")
    plt.figure(figsize=(12, 8))

    T = trajectories.shape[1]
    cmap = plt.cm.inferno_r  
    alpha_min, alpha_max = 0.05, 0.35

    # Batch all segments for each timestep using `None` separators so matplotlib
    # draws many independent line segments with a single plot() call.
    for t in range(T - 1):
        xlist = []
        ylist = []
        for traj in trajectories:
            xlist.extend((traj[t, 0], traj[t + 1, 0], None))
            ylist.extend((traj[t, 1], traj[t + 1, 1], None))

        u = t / (T - 1)                         # 0..1
        alpha = alpha_min + u * (alpha_max - alpha_min)

        plt.plot(
            xlist,
            ylist,
            color=cmap(t / T),
            alpha=alpha
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectories (colored by time)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid()
    plt.tight_layout()
    os.makedirs("./visualizations", exist_ok=True)
    plt.savefig(f"./visualizations/{name}", dpi=500)
    print(f"Saved trajectory plot to ./visualizations/{name}")

def cluster_trajectories(trajectories, n_clusters=1024):
    print(f"Clustering {len(trajectories)} trajectories into {n_clusters} clusters...")
    N, T, D = trajectories.shape
    traj_flat = trajectories.reshape(N, T * D)  # (N, 40)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        random_state=42,
        verbose=2,
    )
    kmeans.fit(traj_flat)

    print("Clustering complete.")
    return kmeans.cluster_centers_.reshape(n_clusters, T, D)  # (n_clusters, 20, 2)


if __name__ == "__main__":
    all_traj = get_all_trajectories()
    plot_trajectories(all_traj)
    cluster_centers = cluster_trajectories(all_traj, n_clusters=1024)
    plot_trajectories(cluster_centers, name="cluster_centers.png")
    np.save("./vocab.npy", cluster_centers)
