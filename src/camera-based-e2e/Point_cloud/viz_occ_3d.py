import numpy as np
import plotly.graph_objects as go
import argparse

VOX_XY_RANGE = 25.0
VOX_XY_RES   = 0.5
VOX_Z_MIN    = -3.0
VOX_Z_RES    = 0.5

COLORS = {
    0: 'lightgrey',  # free
    1: 'red',        # vehicle
    2: 'blue',       # pedestrian
    3: 'orange',     # cyclist
    4: 'darkgrey',   # road
    5: 'green',      # static
}
NAMES = {0:'free', 1:'vehicle', 2:'pedestrian', 3:'cyclist', 4:'road', 5:'static'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occ_path", type=str, required=True)
    parser.add_argument("--show_free", action="store_true")
    args = parser.parse_args()

    grid = np.load(args.occ_path)
    ix, iy, iz = np.where(grid != 255)
    classes = grid[ix, iy, iz]

    x = VOX_XY_RANGE - ix * VOX_XY_RES
    y = VOX_XY_RANGE - iy * VOX_XY_RES
    z = VOX_Z_MIN    + iz * VOX_Z_RES

    traces = []
    for cls in range(0 if args.show_free else 1, 6):
        mask = classes == cls
        if mask.sum() == 0:
            continue
        traces.append(go.Scatter3d(
            x=x[mask], y=y[mask], z=z[mask],
            mode='markers',
            marker=dict(size=3, color=COLORS[cls], opacity=0.8),
            name=NAMES[cls]
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Occupancy Grid 3D View",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=0, y=0, z=2.5)  # top-down by default
            )
        )
    )
    out = args.occ_path.replace('.npy', '.html')
    fig.write_html(out)
    print(f"Saved to {out} — open in browser")

if __name__ == "__main__":
    main()
