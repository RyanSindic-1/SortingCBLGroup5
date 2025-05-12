# visualize_slider.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from simpy_simulation import DT_RECORD


def build_geometry(layout_df):
    """
    Build belt geometry: two straights and two semicircles.
    Returns:
      geom: dict with Lb, Lr, r, L_loop
      decision_distances: list of distances for Infeed, Scanner, Outfeeds
    """
    # Extract key distances
    d_scan = float(layout_df.loc[
        layout_df['Layout property']=='Distance Infeeds to Scanner', 'Value'
    ].iloc[0])
    d_s2o = float(layout_df.loc[
        layout_df['Layout property']=='Distance Scanner to Outfeeds', 'Value'
    ].iloc[0])
    d_betw = float(layout_df.loc[
        layout_df['Layout property']=='Distance between Outfeeds', 'Value'
    ].iloc[0])
    d_back = float(layout_df.loc[
        layout_df['Layout property']=='Distance Outfeeds to Infeeds', 'Value'
    ].iloc[0])

    # Bottom straight from Infeed (0) to last Outfeed:
    Lb = d_scan + d_s2o + 2 * d_betw
    # For symmetry, top straight = bottom
    Lr = Lb
    # Semicircle radius so that 2*pi*r + Lr = d_back
    r = max((d_back - Lr) / (2 * np.pi), 0.1 * Lb)
    # Total loop length
    L_loop = Lb + 2 * np.pi * r + Lr

    # Decision points along bottom straight:
    # Infeed=0, Scanner=d_scan, Out1=d_scan+d_s2o, Out2=...+d_betw, Out3=...+d_betw
    decision_distances = [0.0,
                          d_scan,
                          d_scan + d_s2o,
                          d_scan + d_s2o + d_betw,
                          d_scan + d_s2o + 2 * d_betw]
    return ({'Lb': Lb, 'Lr': Lr, 'r': r, 'L_loop': L_loop}, decision_distances)


def dist_to_xy(s, geom):
    """
    Map a scalar distance along the loop to (x,y).
    """
    Lb, Lr, r, L_loop = geom['Lb'], geom['Lr'], geom['r'], geom['L_loop']
    s_mod = s % L_loop
    # 1) bottom straight [0, Lb)
    if s_mod < Lb:
        return (s_mod, 0.0)
    s2 = s_mod - Lb
    # 2) right semicircle [Lb, Lb+pi*r)
    if s2 < np.pi * r:
        theta = -np.pi/2 + s2 / r
        return (Lb + r * np.cos(theta), r + r * np.sin(theta))
    s3 = s2 - np.pi * r
    # 3) top straight [.., ..+Lr)
    if s3 < Lr:
        return (Lb - s3, 2 * r)
    # 4) left semicircle
    s4 = s3 - Lr
    theta = np.pi/2 + s4 / r
    return (r * np.cos(theta), r + r * np.sin(theta))


def main():
    # Load data
    layout_df = pd.ExcelFile('PosiSorterData1.xlsx').parse('Layout')
    pos_df = pd.read_csv('positions.csv')

    # Build geometry and decision points
    geom, decision_distances = build_geometry(layout_df)
    decision_labels = ['Infeed', 'Scanner', 'Outfeed 1', 'Outfeed 2', 'Outfeed 3']
    decision_xy = np.array([dist_to_xy(d, geom) for d in decision_distances])

    # Group positions by time for fast lookup
    times = np.sort(pos_df['time_s'].unique())
    max_time = times[-1]
    pos_map = {t: pos_df.loc[pos_df['time_s']==t, 'dist_m'].values for t in times}

    # Figure setup
    fig, ax = plt.subplots(figsize=(8,6))
    # Draw belt outline
    sample_s = np.linspace(0, geom['L_loop'], 400)
    belt_xy = np.array([dist_to_xy(s, geom) for s in sample_s])
    ax.plot(belt_xy[:,0], belt_xy[:,1], '-k', lw=2)
    # Draw decision points
    ax.scatter(decision_xy[:,0], decision_xy[:,1], c='red', s=80, zorder=5)
    for (x,y), lbl in zip(decision_xy, decision_labels):
        ax.text(x, y - 0.1 * geom['r'], lbl, ha='center', va='top', fontsize=9)

    # Parcel scatter
    scat = ax.scatter([], [], s=60, edgecolors='blue', facecolors='cyan', zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('PosiSorter Conveyor Simulation')

    # Slider for speed
    slider_ax = fig.add_axes([0.25, 0.02, 0.5, 0.03])
    speed_slider = Slider(slider_ax, 'Speed', 0.1, 15.0, valinit=3.0)

    # Simulation time state
    sim_time = 0.0

    def update(frame):
        nonlocal sim_time
        # advance simulation time by DT_RECORD * speed
        sim_time += DT_RECORD * speed_slider.val
        # loop
        if sim_time > max_time:
            sim_time -= max_time
        # find nearest recorded time
        idx = int(round(sim_time / DT_RECORD))
        t = times[min(idx, len(times)-1)]
        dists = pos_map.get(t, np.array([]))
        coords = np.array([dist_to_xy(d, geom) for d in dists]) if dists.size else np.empty((0,2))
        scat.set_offsets(coords)
        return scat,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),            # explicitly set number of frames
        interval=DT_RECORD * 1000,   # interval in milliseconds
        blit=True,
        cache_frame_data=False       # disable unbounded frame caching
    )
