# visualize_numeric.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Settings: adjust these if you like ---
DOWNSAMPLE_FACTOR = 5    # keep 1 in every 5 frames
PARCEL_MARKER_SIZE = 60
LINE_WIDTH = 2

# --- Geometry builders ---
def read_geometry(xlsx="PosiSorterData1.xlsx"):
    df = pd.ExcelFile(xlsx).parse("Layout")
    d_scan    = float(df.loc[df["Layout property"]=="Distance Infeeds to Scanner","Value"].iloc[0])
    d_s2o     = float(df.loc[df["Layout property"]=="Distance Scanner to Outfeeds","Value"].iloc[0])
    d_between = float(df.loc[df["Layout property"]=="Distance between Outfeeds","Value"].iloc[0])
    d_back    = float(df.loc[df["Layout property"]=="Distance Outfeeds to Infeeds","Value"].iloc[0])

    # bottom straight Lb and return straight Lr
    Lb = d_scan + d_s2o + 2*d_between
    Lr = Lb
    # semicircle radius so 2πr + Lr = d_back
    r = max((d_back - Lr)/(2*np.pi), 0.1*Lb)
    L  = Lb + 2*np.pi*r + Lr

    # decision‐point distances along the bottom straight
    dp = [
        0.0,                       # Infeed
        d_scan,                    # Scanner
        d_scan + d_s2o,            # Outfeed 1
        d_scan + d_s2o + d_between,# Outfeed 2
        d_scan + d_s2o + 2*d_between # Outfeed 3
    ]

    return {"Lb":Lb, "Lr":Lr, "r":r, "L":L}, dp

def dist_to_xy(s, geom):
    """Map float distance along the loop to (x,y)."""
    Lb, Lr, r, L = geom["Lb"], geom["Lr"], geom["r"], geom["L"]
    m = s % L
    # bottom straight
    if m < Lb:
        return (m, 0.0)
    m -= Lb
    # right semicircle
    if m < np.pi*r:
        θ = -np.pi/2 + m/r
        return (Lb + r*np.cos(θ), r + r*np.sin(θ))
    m -= np.pi*r
    # top straight (going left)
    if m < Lr:
        return (Lb - m, 2*r)
    m -= Lr
    # left semicircle
    θ = np.pi/2 + m/r
    return (r*np.cos(θ), r + r*np.sin(θ))

# --- Main visualization ---
def main():
    # 1) Load geometry + decision points
    geom, dp_dist = read_geometry()
    dp_xy = np.array([dist_to_xy(d, geom) for d in dp_dist])
    dp_labels = ["Infeed","Scanner","Outfeed 1","Outfeed 2","Outfeed 3"]

    # 2) Load and downsample positions.csv
    pos = pd.read_csv("positions.csv")  # columns: time_s, parcel_id, dist_m
    times = np.sort(pos["time_s"].unique())
    times = times[::DOWNSAMPLE_FACTOR]  # keep 1 in N frames
    pos = pos[pos["time_s"].isin(times)]

    # 3) Pre‐group distances by frame for fast lookup
    grouped = {t: grp["dist_m"].values for t, grp in pos.groupby("time_s")}

    # 4) Prepare plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PosiSorter 2D Conveyor (Numeric times)")

    # draw belt outline
    sample_s = np.linspace(0, geom["L"], 400)
    outline = np.array([dist_to_xy(s, geom) for s in sample_s])
    ax.plot(outline[:,0], outline[:,1], "-k", lw=LINE_WIDTH)

    # draw decision points
    ax.scatter(dp_xy[:,0], dp_xy[:,1], c="red", s=80, zorder=5)
    for (x,y),lbl in zip(dp_xy, dp_labels):
        ax.text(x, y - 0.05*geom["r"], lbl, ha="center", va="top")

    # parcel scatter
    scat = ax.scatter([], [], s=PARCEL_MARKER_SIZE,
                      edgecolors="blue", facecolors="cyan", zorder=10)

    # 5) Animation update
    def update(frame_idx):
        t = times[frame_idx]
        dists = grouped.get(t, np.array([]))
        coords = np.array([dist_to_xy(d, geom) for d in dists]) if dists.size else np.empty((0,2))
        scat.set_offsets(coords)
        return scat,

    ani = FuncAnimation(
        fig, update,
        frames=len(times),
        interval=DT_RECORD*1000*DOWNSAMPLE_FACTOR,  # ms per frame
        blit=True
    )

    plt.show()

if __name__=="__main__":
    # import DT_RECORD from your simulation module (seconds)
    from simpy_simulation import DT_RECORD
    main()
