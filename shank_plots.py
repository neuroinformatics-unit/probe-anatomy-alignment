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
    tip_size_um: float = 0.0,
    initial_offset_um: float = 0.0,
    step_um: float = 25.0,
    title: str = "Shank Alignment",
):
    """
    Interactively nudge channels (and optional PSD power trace) up/down relative
    to atlas tracks using Matplotlib buttons.

    Parameters
    ----------
    tracks : pandas.DataFrame
        Must contain columns:
          - "Distance from first position [um]" (strictly increasing, starts at 0)
          - "Region name"
    channel_locs : (N, 2) array-like
        Channel XY positions in microns (SpikeInterface-style). y=0 near tip.
    power_per_channel : (N,) array-like or None
        Optional per-channel power to draw as a ribbon along the channels.
    tip_size_um : float
        Length of the probe tip to subtract when mapping to track distance.
    initial_offset_um : float
        Starting vertical offset (µm). Positive moves channels deeper.
    step_um : float
        Amount (µm) to nudge per button press.
    title : str
        Figure title prefix.

    Returns
    -------
    probe_offset_um : float
        The final offset after clicking "Done" and closing the window.
    """
    # ---- basic extraction ----
    track_pos = np.asarray(tracks["Distance from first position [um]"].to_numpy())
    regions = tracks["Region name"].astype(str).replace({"nan": "Unknown"}).to_numpy()

    if not np.all(np.diff(track_pos) > 0):
        raise ValueError("tracks['Distance from first position [um]'] must be strictly increasing.")
    if track_pos[0] != 0:
        raise ValueError("tracks must start at 0 µm distance from first position.")

    y_min, y_max = float(track_pos.min()), float(track_pos.max())
    seg_end = y_max  # last distance along the segmented track

    # Identify region blocks (runs of the same region)
    change_idx = np.where(regions[:-1] != regions[1:])[0] + 1
    region_bounds = np.concatenate(([0], change_idx, [len(track_pos)]))

    # ---- interactive state ----
    state = {"offset": float(initial_offset_um), "done": False}

    # ---- figure & layout ----
    fig, ax = plt.subplots(figsize=(7, 9))
    # Make room for right-side labels and bottom buttons
    plt.subplots_adjust(right=0.86, bottom=0.18)

    # ---- core drawing routine ----
    def draw_plot():
        ax.clear()

        # --- shaded region spans ---
        colors = plt.cm.tab20.colors
        for i in range(len(region_bounds) - 1):
            start = track_pos[region_bounds[i]]
            end   = track_pos[region_bounds[i + 1] - 1]
            ax.axhspan(start, end, color=colors[i % len(colors)], alpha=0.3)

        # --- right-side region labels (de-overlapped a bit) ---
        label_positions = []
        label_names = []
        for i in range(len(region_bounds) - 1):
            start = track_pos[region_bounds[i]]
            end   = track_pos[region_bounds[i + 1] - 1]
            label_positions.append(0.5 * (start + end))
            label_names.append(regions[region_bounds[i]])

        if len(label_positions) > 1:
            label_positions = np.array(label_positions, float)
            # greedy separation to avoid collisions
            sort_idx = np.argsort(label_positions)
            label_positions = label_positions[sort_idx]
            label_names = np.array(label_names, dtype=object)[sort_idx]
            min_sep = max(10.0, np.diff(track_pos).min() * 2)
            for k in range(1, len(label_positions)):
                if label_positions[k] - label_positions[k - 1] < min_sep:
                    label_positions[k] = label_positions[k - 1] + min_sep

        for pos, name in zip(label_positions, label_names):
            ax.text(
                1.02, pos, str(name),
                transform=ax.get_yaxis_transform(),  # x in axes-fraction, y in data coords
                ha="left", va="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                clip_on=False, zorder=5
            )

        # --- segmentation points (left rail) ---
        ax.scatter(np.zeros_like(track_pos), track_pos, s=20, color='black', zorder=3)

        # --- channels mapped into track space with current offset ---
        chan = np.array(channel_locs, float).copy()
        # map y to atlas distance along the track, accounting for offset and tip size
        chan[:, 1] = seg_end - tip_size_um - (chan[:, 1] + state["offset"])

        # normalize x into a slim column near ~0.25 for visibility
        xs = chan[:, 0]
        if np.ptp(xs) == 0:
            chan[:, 0] = 0.25
        else:
            chan[:, 0] = ((xs - xs.min()) / np.ptp(xs)) * 0.05 + 0.25

        ax.scatter(chan[:, 0], chan[:, 1], color='red', marker='x', s=35, zorder=4, label="Channels")

        # --- optional power ribbon (normalized to [0,1] mapped into x in [0.5, 1.0]) ---
        if power_per_channel is not None:
            p = np.asarray(power_per_channel, dtype=float)
            if p.ndim != 1 or p.shape[0] != chan.shape[0]:
                raise ValueError("power_per_channel must be 1D with one value per channel.")
            pmin, pmax = np.nanmin(p), np.nanmax(p)
            p_norm = np.zeros_like(p) if np.isclose(pmax - pmin, 0) else (p - pmin) / (pmax - pmin)
            x_power = 0.5 + p_norm * 0.5
            order = np.argsort(chan[:, 1])  # continuous ribbon from shallow->deep
            ax.plot(x_power[order], chan[:, 1][order], lw=1.5, color="black", label="Power (norm)")

        # --- axes cosmetics & locking ---
        range_ = y_max - y_min
        padding = range_ * 0.25
        ax.set_ylim(y_min - padding, y_max + padding)   # lock absolute scale (no visual shrinking)
        ax.invert_yaxis()           # deeper = larger distance
        ax.set_ylabel("Distance from first position [μm]")
        ax.set_title(f"{title}  |  offset = {state['offset']:.1f} µm")
        ax.set_xticks([0.0, 0.25, 1.0])
        ax.set_xticklabels(["Tracks", "Channels", "Power max"], rotation=30)
        ax.grid(False)

        fig.canvas.draw_idle()

    # ---- callbacks ----
    def on_up(_):
        state["offset"] += step_um
        draw_plot()

    def on_down(_):
        state["offset"] -= step_um
        draw_plot()

    def on_done(_):
        state["done"] = True
        print(f"Final probe_offset_um = {state['offset']:.1f} µm")
        plt.close(fig)

    # Optional: arrow keys for nudging
    def on_key(ev):
        if ev.key in ("up", "w"):
            on_up(None)
        elif ev.key in ("down", "s"):
            on_down(None)
        elif ev.key in ("enter",):
            on_done(None)

    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # ---- small buttons (no emoji to avoid font warnings) ----
    # [left, bottom, width, height] in figure fractions
    ax_down = plt.axes([0.16, 0.05, 0.10, 0.05])
    ax_up   = plt.axes([0.28, 0.05, 0.10, 0.05])
    ax_done = plt.axes([0.74, 0.05, 0.12, 0.05])

    btn_down = Button(ax_down, f"▼ -{step_um:.0f}µm")
    btn_up   = Button(ax_up,   f"▲ +{step_um:.0f}µm")
    btn_done = Button(ax_done, "Done")

    for b in (btn_down, btn_up, btn_done):
        b.label.set_fontsize(8)

    btn_up.on_clicked(on_up)
    btn_down.on_clicked(on_down)
    btn_done.on_clicked(on_done)

    # ---- initial render & show ----
    draw_plot()
    plt.show()

    fig.canvas.mpl_disconnect(cid)
    return state["offset"]
