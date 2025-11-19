import matplotlib
# Set non-interactive backend to save files without needing a window system
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loader import WaymoE2E
from torch.utils.data import DataLoader

# Setup plotting style
sns.set_theme(style="whitegrid")

dataset = WaymoE2E(
    batch_size=30000, 
    data_dir="/scratch/gilbreth/bnamikas/data/waymo_open_dataset_end_to_end_camera_v_1_0_0", 
    images=False, 
    n_items=30000
)

loader = DataLoader(
    dataset, 
    batch_size=30000,
    num_workers=1,
)
    
for data in loader:
    # --- Data Extraction ---
    past = data['PAST'].numpy()   # (B, 16, 6)
    future = data['FUTURE'].numpy() # (B, 16, 2)
    
    intent = None
    if 'INTENT' in data:
        intent = data['INTENT'].numpy().flatten()
    elif 'intent' in data:
        intent = data['intent'].numpy().flatten()

    print(f"Processing Batch - Future Shape: {future.shape}")
    
    # CONFIG FOR "MANY MORE" PLOTS
    # Use all data in the batch
    num_samples = len(past) 
    # Make lines very faint and thin so they stack nicely
    traj_alpha = 0.05  
    traj_width = 0.5

    # --- 1. Global Kinematics (Velocity & Acceleration) ---
    print("Generating Global Kinematics plot...")
    fig_kin, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    vel_x = past[:, :, 2].flatten()
    vel_y = past[:, :, 3].flatten()
    sns.kdeplot(vel_x, fill=True, label='Vel X', color='tab:blue', alpha=0.3, ax=axes[0])
    sns.kdeplot(vel_y, fill=True, label='Vel Y', color='tab:orange', alpha=0.3, ax=axes[0])
    axes[0].set_title("Global Velocity Distribution")
    axes[0].legend()

    acc_x = past[:, :, 4].flatten()
    acc_y = past[:, :, 5].flatten()
    sns.kdeplot(acc_x, fill=True, label='Accel X', color='tab:green', alpha=0.3, ax=axes[1])
    sns.kdeplot(acc_y, fill=True, label='Accel Y', color='tab:red', alpha=0.3, ax=axes[1])
    axes[1].set_title("Global Acceleration Distribution")
    axes[1].legend()

    speed = np.sqrt(vel_x**2 + vel_y**2)
    sns.histplot(speed, kde=True, color='tab:purple', ax=axes[2], bins=50)
    axes[2].set_title("Global Speed Magnitude Distribution")
    
    plt.tight_layout()
    fig_kin.savefig('global_kinematics.png')
    plt.close(fig_kin)

    # --- 2. Velocity Snapshots ---
    print("Generating Velocity Snapshot analysis...")
    fig_vel, ax_vel = plt.subplots(1, 2, figsize=(16, 6))

    vel_x_0 = past[:, -1, 2]
    vel_y_0 = past[:, -1, 3]
    speed_t0 = np.sqrt(vel_x_0**2 + vel_y_0**2)

    dt = 0.25
    last_step_disp = future[:, -1, :] - future[:, -2, :]
    speed_t4 = np.sqrt(last_step_disp[:, 0]**2 + last_step_disp[:, 1]**2) / dt

    sns.histplot(speed_t0, color='teal', alpha=0.5, label='t=0', kde=True, ax=ax_vel[0])
    sns.histplot(speed_t4, color='maroon', alpha=0.5, label='t=4', kde=True, ax=ax_vel[0])
    ax_vel[0].set_title("Speed Distribution (t=0 vs t=4)")
    ax_vel[0].legend()

    speed_delta = speed_t4 - speed_t0
    sns.histplot(speed_delta, color='orange', kde=True, ax=ax_vel[1])
    ax_vel[1].axvline(0, color='black', linestyle='--')
    ax_vel[1].set_title("Delta Speed (t=4 - t=0)")

    plt.tight_layout()
    fig_vel.savefig('velocity_snapshots.png')
    plt.close(fig_vel)

    # --- 3. X-Displacement Analysis ---
    print("Generating Future X vs Initial X analysis...")
    fig_x, ax_x = plt.subplots(1, 2, figsize=(16, 6))

    initial_x = past[:, -1, 0]
    final_x = future[:, -1, 0]

    sns.scatterplot(x=initial_x, y=final_x, alpha=0.3, s=10, ax=ax_x[0], hue=intent, palette='viridis' if intent is not None else None, linewidth=0)
    min_val = min(initial_x.min(), final_x.min())
    max_val = max(initial_x.max(), final_x.max())
    ax_x[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax_x[0].set_title("Scatter: Final Future X vs Initial X")
    
    x_displacement = final_x - initial_x
    if intent is not None:
        sns.histplot(x=x_displacement, hue=intent, kde=True, palette='viridis', ax=ax_x[1], element="step")
    else:
        sns.histplot(x_displacement, kde=True, ax=ax_x[1], color='purple')
    ax_x[1].axvline(0, color='black', linestyle='--')
    ax_x[1].set_title("X-Displacement Histogram")

    plt.tight_layout()
    fig_x.savefig('x_displacement_analysis.png')
    plt.close(fig_x)
    
    # --- 4. Phase & Jerk (Existing) ---
    # (Condensed for brevity, same logic as before)
    fig_phase, ax_phase = plt.subplots(figsize=(10, 8))
    flat_vx = past[:, :, 2].flatten()
    flat_vy = past[:, :, 3].flatten()
    flat_ax = past[:, :, 4].flatten()
    flat_ay = past[:, :, 5].flatten()
    flat_speed = np.sqrt(flat_vx**2 + flat_vy**2)
    epsilon = 1e-6
    flat_speed_safe = flat_speed.copy()
    flat_speed_safe[flat_speed_safe < epsilon] = epsilon
    long_accel = (flat_vx * flat_ax + flat_vy * flat_ay) / flat_speed_safe
    move_mask = flat_speed > 0.1
    sns.kdeplot(x=flat_speed[move_mask], y=long_accel[move_mask], fill=True, cmap="rocket_r", thresh=0.05, levels=15, ax=ax_phase)
    ax_phase.set_title("Phase Portrait (Speed vs. Long. Accel)")
    fig_phase.savefig('phase_portrait.png')
    plt.close(fig_phase)
    
    fig_jerk, ax_jerk = plt.subplots(1, 2, figsize=(16, 6))
    jerk_x = np.diff(past[:, :, 4], axis=1) / 0.25
    jerk_y = np.diff(past[:, :, 5], axis=1) / 0.25
    jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2).flatten()
    sns.histplot(jerk_mag, kde=True, color='crimson', ax=ax_jerk[0], bins=50, log_scale=(False, True))
    if intent is not None:
        intent_expanded = np.repeat(intent[:, np.newaxis], jerk_x.shape[1], axis=1).flatten()
        sns.boxplot(x=intent_expanded, y=jerk_mag, palette='viridis', ax=ax_jerk[1], showfliers=False)
    fig_jerk.savefig('jerk_distribution.png')
    plt.close(fig_jerk)

    # --- 5. MASSIVE TRAJECTORY PLOTS ---
    if intent is not None:
        print("Generating High-Density Trajectory plots...")
        
        # A. Global Trajectories (ALL LINES)
        fig_glob_traj, ax_glob = plt.subplots(figsize=(12, 12))
        
        colors = {1: 'blue', 2: 'green', 3: 'red', 0: 'grey'}
        
        # Loop through ALL indices, not just a sample
        # We iterate by intent group to minimize plt.plot calls (optimization)
        unique_intents = np.unique(intent)
        for k in unique_intents:
            mask = (intent == k)
            # Extract relative future paths for this intent
            future_k = future[mask]
            past_k = past[mask]
            
            # Calculate relative to current pos
            # future_k shape: (N, 16, 2)
            # current pos: (N, 1, 2) -> taken from last step of past
            curr_pos = past_k[:, -1, :2][:, np.newaxis, :]
            rel_future = future_k - curr_pos
            
            # To plot efficiently, we can use LineCollection or just loop if N < 5000
            # With N=1000, looping is "okay" but slow. 
            # Let's use a simpler approach: Transpose and plot chunks
            # Plotting 1000 lines with low alpha
            c = colors.get(int(k), 'grey')
            
            # Vectorized plot trick: Plot all segments separated by NaN? 
            # Or just loop. For 1000 items, loop is acceptable in offline script.
            for i in range(len(rel_future)):
                ax_glob.plot(rel_future[i, :, 0], rel_future[i, :, 1], color=c, alpha=traj_alpha, linewidth=traj_width)
            
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[k], lw=2, label=f'Intent {k}') for k in sorted(unique_intents)]
        ax_glob.legend(handles=legend_elements)
        ax_glob.set_title(f"Global Future Trajectories (All {num_samples} Vehicles)")
        ax_glob.set_xlabel("Relative X (m)")
        ax_glob.set_ylabel("Relative Y (m)")
        ax_glob.axis('equal')
        
        plt.tight_layout()
        fig_glob_traj.savefig('global_trajectories_high_density.png')
        plt.close(fig_glob_traj)

        # B. Lane Reconstruction (2D Histogram of ALL trajectory points)
        # This reveals the "Map" without having the map
        fig_map, ax_map = plt.subplots(figsize=(12, 12))
        
        # Collect ALL points from PAST trajectories (relative to t=0)
        # We normalize everyone to start at (0,0) to see RELATIVE motion patterns
        # OR we plot absolute coords if we want to see the map structure (if data is absolute)
        # Assuming data is absolute world coords:
        all_x = past[:, :, 0].flatten()
        all_y = past[:, :, 1].flatten()
        
        # Use hist2d to create a heatmap of road occupancy
        h = ax_map.hist2d(all_x, all_y, bins=200, cmap='inferno', norm=matplotlib.colors.LogNorm())
        fig_map.colorbar(h[3], ax=ax_map, label='Log Count of Points')
        
        ax_map.set_title("Lane Reconstruction (Global Position Density)")
        ax_map.set_xlabel("Global X")
        ax_map.set_ylabel("Global Y")
        ax_map.axis('equal')
        fig_map.savefig('global_lane_density.png')
        plt.close(fig_map)

        # C. Per-Intent High Density Plots
        for k in unique_intents:
            print(f"Generating High-Density analysis for Intent {k}...")
            
            mask = (intent == k)
            past_k = past[mask]
            future_k = future[mask]
            if len(past_k) == 0: continue

            fig_k, ax_k = plt.subplots(1, 2, figsize=(20, 8))
            
            # 1. High Density Lines
            curr_pos = past_k[:, -1, :2][:, np.newaxis, :]
            rel_past = past_k[:, :, :2] - curr_pos
            rel_fut = future_k - curr_pos
            
            # Plotting ALL lines for this intent
            for i in range(len(rel_past)):
                # Plot Past (Grey)
                ax_k[0].plot(rel_past[i, :, 0], rel_past[i, :, 1], color='grey', alpha=traj_alpha, linewidth=traj_width)
                # Plot Future (Blue)
                ax_k[0].plot(rel_fut[i, :, 0], rel_fut[i, :, 1], color='blue', alpha=traj_alpha, linewidth=traj_width)
            
            ax_k[0].set_title(f"Intent {k}: All {len(past_k)} Trajectories")
            ax_k[0].axis('equal')
            
            # 2. Path Density (Heatmap of the entire LINE, not just endpoints)
            # Flatten relative future coordinates
            flat_fx = rel_fut[:, :, 0].flatten()
            flat_fy = rel_fut[:, :, 1].flatten()
            
            sns.kdeplot(x=flat_fx, y=flat_fy, fill=True, cmap="Blues", ax=ax_k[1], thresh=0.05, levels=20)
            ax_k[1].scatter(0, 0, marker='+', color='black', s=100, label='Current Pos')
            ax_k[1].set_title(f"Intent {k}: Future Path Probabilities (KDE)")
            ax_k[1].axis('equal')

            plt.suptitle(f"High-Density Analysis: Intent Class {k}", fontsize=16)
            plt.tight_layout()
            plt.savefig(f'analysis_intent_{int(k)}_high_density.png')
            plt.close(fig_k)

    break