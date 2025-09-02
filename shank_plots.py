import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import bisect

from h5py.h5pl import insert


def compute_chunk_start_samples(num_segments, nperseg, n_samples, max_iter=1000):
    """
    """
    all_start_samples = []
    attempts = 0

    while len(all_start_samples) < num_segments:

        if attempts > max_iter:
            raise RuntimeError("Too many attempts; maybe num_segments is too large for n_samples.")
        attempts += 1

        putative_start = np.random.randint(0, n_samples - nperseg + 1)
        insert_idx = bisect.bisect_left(all_start_samples, putative_start)

        # Check overlap with previous segment
        if insert_idx > 0 and putative_start < all_start_samples[insert_idx - 1] + nperseg:
            continue

        # Check overlap with next segment
        if insert_idx < len(all_start_samples) and all_start_samples[insert_idx] < putative_start + nperseg:
            continue

        # Safe to insert while keeping list sorted
        all_start_samples.insert(insert_idx, putative_start)

    return all_start_samples


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

def interactive_probe_alignment(
    tracks,
    channel_locs,
    power_per_channel=None,
    tip_size_um=0.0,
    initial_offset_um=0.0,
    step_um=10.0,
    title="Shank Alignment"
):
    """
    Interactive plot: move channels + PSD trace up/down using Matplotlib buttons.
    Returns the final probe_offset_um.
    """

    # Keep state in a dict so inner functions can modify it
    state = {"offset": initial_offset_um}

    # Create the figure + axis
    fig, ax = plt.subplots(figsize=(7, 9))
    plt.subplots_adjust(bottom=0.15)  # Make room for buttons

    def draw_plot():
        ax.clear()

        track_pos = tracks["Distance from first position [um]"].to_numpy()
        regions = tracks["Region name"].astype(str).replace({"nan": "Unknown"}).to_numpy()

        # --- shifted channels ---
        seg_end = track_pos.max()
        chan_pos_on_track = channel_locs.copy()
        chan_pos_on_track[:, 1] = seg_end - tip_size_um - (chan_pos_on_track[:, 1] + state["offset"])

        # ---- shaded regions ----
        change_idx = np.where(regions[:-1] != regions[1:])[0] + 1
        region_bounds = np.concatenate(([0], change_idx, [len(track_pos)]))
        color_cycle = plt.cm.tab20.colors
        for i in range(len(region_bounds) - 1):
            start = track_pos[region_bounds[i]]
            start = track_pos[region_bounds[i]]
            end = track_pos[region_bounds[i + 1] - 1]
            color = color_cycle[i % len(color_cycle)]
            ax.axhspan(start, end, color=color, alpha=0.3)

        # ---- scatter channels ----
        xs = chan_pos_on_track[:, 0]
        if np.ptp(xs) == 0:
            chan_pos_on_track[:, 0] = 0.25
        else:
            chan_pos_on_track[:, 0] = ((xs - xs.min()) / np.ptp(xs)) * 0.05 + 0.25
        ax.scatter(np.zeros_like(track_pos), track_pos, s=20, color="black")
        ax.scatter(chan_pos_on_track[:, 0], chan_pos_on_track[:, 1], color="red", marker="x", s=35)

        # ---- PSD trace ----
        if power_per_channel is not None:
            p = np.asarray(power_per_channel, dtype=float)
            pmin, pmax = np.nanmin(p), np.nanmax(p)
            p_norm = np.zeros_like(p) if np.isclose(pmax - pmin, 0) else (p - pmin) / (pmax - pmin)
            x_power = 0.5 + p_norm * 0.5
            order = np.argsort(chan_pos_on_track[:, 1])
            ax.plot(x_power[order], chan_pos_on_track[:, 1][order], lw=1.5, color="black")

        # --- lock axis scaling ---
        track_pos_range = track_pos.max() - track_pos.min()
        y_axis_pad = track_pos_range * 0.25

        ax.set_ylim(track_pos.min() - y_axis_pad, track_pos.max() + y_axis_pad)

        ax.set_ylabel("Distance from first position [μm]")
        ax.set_title(f"{title} | offset = {state['offset']:.1f} µm")
        ax.invert_yaxis()
        ax.set_xticks([0, 0.25, 1.0])
        ax.set_xticklabels(["Tracks", "Channels", "Power max"], rotation=30)

        fig.canvas.draw_idle()

    # ---- button callbacks ----
    def on_up(event):
        state["offset"] += step_um
        draw_plot()

    def on_down(event):
        state["offset"] -= step_um
        draw_plot()

    def on_done(event):
        print(f"✅ Final probe_offset_um = {state['offset']:.1f} µm")
        plt.close(fig)

    # ---- create buttons ----
    ax_up = plt.axes([0.15, 0.02, 0.15, 0.02])   # x, y, width, height
    ax_down = plt.axes([0.35, 0.02, 0.15, 0.02])
    ax_done = plt.axes([0.70, 0.02, 0.15, 0.02])

    btn_up = Button(ax_up, f"Up +{step_um:.0f} µm")
    btn_down = Button(ax_down, f"Down -{step_um:.0f} µm")
    btn_done = Button(ax_done, "Done")

    btn_up.on_clicked(on_up)
    btn_down.on_clicked(on_down)
    btn_done.on_clicked(on_done)

    # ---- initial draw ----
    draw_plot()
    plt.show()

    return state
