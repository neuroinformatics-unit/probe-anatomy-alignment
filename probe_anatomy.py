from spikeinterface.extractors import read_spikeglx
from pathlib import Path
import pandas as pd
import numpy as np
import bisect
from scipy import signal
from spikeinterface.preprocessing.tests.test_zero_padding import recording
import matplotlib.pyplot as plt
from shank_plots import interactive_probe_alignment, compute_chunk_start_samples

# brainglobe, distances are closest to the surface  TODO: CHECK THIS IS TRUE! is it the surface or first position?


# TODO:
# 1) try with large file
# 2) add peak detection method
# 3) thoroughly check and send to Tilly, read those 2x papers
# 4) will need to finally report something that gets the units locations, but meet with Tilly first to check
# 5) double check preprocessing that needs to be done (I dont think any)

offset_detection = "manual" # "manual" or "peak_detect"

# 2) check through `plot_tracks_with_regions`
# 3) ensure FFT is in the correct order
# 4) compute on full data, replicate expected FFT
# 5) check the output thoroughly and the Barletts thoroughly
# 6) Add a little widget to move it up and down, also compute based on peak
# 7) figure out what the exact output should be

base_path = Path(r"C:\Users\Jzimi\Downloads\probe-segmentation\probe-segmentation")
tip_size_um = 175

tracks_to_shank = {
    0: "track_1.csv",
    1: "track_2.csv",
}

# Read Channel Locations

shanks = {}

raw_recording =  read_spikeglx(base_path)
recordings_split = raw_recording.split_by("group")

# Welch's method parameters

n_samples = raw_recording.get_num_samples()
fs = raw_recording.get_sampling_frequency()

num_segments = 10
nperseg = int(fs) if fs < n_samples / num_segments else 2000
hann_window = signal.windows.hann(nperseg, sym=False)  # TODO: CHECK

max_iter = 1000
all_start_samples = []
attempts = 0

lower_hz = 500
upper_hz = 1250

all_start_samples = compute_chunk_start_samples(num_segments, nperseg, n_samples)

for idx, rec in recordings_split.items():

    shanks[idx]  = {}

    # Load the channel locations from spikeinterface + the brainglobe segmentation tracks
    channel_locs = rec.get_channel_locations()

    shanks[idx]["channel_locs"] = channel_locs

    tracks = pd.read_csv(base_path / "segmentation" / "atlas_space" / "tracks" / tracks_to_shank[idx])

    shanks[idx]["tracks"] = tracks

    # In spikeinterface, channels at depth y = 0 are closest to the tip. In
    # brainglobe, distances are closest to the first clicked point during segmentation,
    # which should be the surfrace. So we must find the position of the channel on the
    # segmentation, accounting for the tip size. Because the distance between track points
    # (spline interpolated) are not all the same, `np.searchsorted` must be used.

    track_pos = tracks["Distance from first position [um]"].to_numpy()

    assert not np.any(np.diff(track_pos) <= 0), "track positions must be strictly increasing"
    assert track_pos[0] == 0, "track positions must start at zero"

    seg_end = np.max(track_pos)

    channel_locs_y = channel_locs[:, 1]

    chan_to_track_pos = seg_end - tip_size_um - channel_locs_y

    shanks[idx]["chan_to_track_pos"] = chan_to_track_pos
    shanks[idx]["chan_to_track_idx"] = np.searchsorted(track_pos, chan_to_track_pos)
    shanks[idx]["chan_to_brain_region"] = tracks["Region name"][shanks[idx]["chan_to_track_idx"]]

    # Compute the PSD using Windowed Bartlett methods

    psd = np.zeros((np.ceil(nperseg / 2).astype(int) + 1, rec.get_num_channels()))  # TODO: check

    for start_sample in all_start_samples:
        chunk = rec.get_traces(start_frame=start_sample, end_frame=start_sample + nperseg, return_scaled=True)
        psd[:, :] += np.abs(np.fft.rfft(chunk * hann_window[:, np.newaxis], axis=0)*2)**2 * (1 / num_segments) # *  window_power_norm, also check the doubling
        # consider div by U
        # dont double DC bins

    fft_bins = np.fft.rfftfreq(nperseg, 1 / fs)

    shanks[idx]["psd"] = psd
    shanks[idx]["psd_freqs"] = fft_bins

    lower_hz_idx =  np.searchsorted(fft_bins, lower_hz)
    upper_hz_idx = np.searchsorted(fft_bins, upper_hz)

    shanks[idx]["power_per_channel"] = np.mean(psd[lower_hz_idx:upper_hz_idx + 1, :], axis=0)  # TODO: check the upper bound

    if offset_detection == "manual":

        state = interactive_probe_alignment(
            tracks=tracks,
            channel_locs=channel_locs,
            power_per_channel=shanks[idx]["power_per_channel"],  # or None
            tip_size_um=tip_size_um,
            initial_offset_um=0.0,
            step_um=25.0,
            title=f"Shank {idx}: Alignment"
        )
        shanks[idx]["probe_offset_um"] = state["offset"]  # Check is um
    else:
        raise NotImplementedError()

