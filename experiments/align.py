from __future__ import annotations

import sys

if "." not in sys.path:
    sys.path.append(".")

import argparse
import dataclasses
import logging
import pathlib
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
import tqdm

from model.model import *  # noqa: F403

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path_to_model: str) -> torch.nn.Module:
    logging.info(f"Loading model {path_to_model}...")
    raw_model = torch.load(path_to_model, map_location=DEVICE)
    arch = raw_model["arch"]

    try:
        model_type = raw_model["model"]
    except KeyError:
        model_type = raw_model["config"]["model"]

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model["state_dict"])

    return model


@dataclasses.dataclass
class Args:
    input_events: pathlib.Path
    input_video: pathlib.Path

    @classmethod
    def from_cli(cls) -> Args:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_events",
            required=True,
            type=pathlib.Path,
            help="Path to the input events .bin file",
        )
        parser.add_argument(
            "--input_video",
            required=True,
            type=pathlib.Path,
            help="Path to the input video .mp4 file",
        )
        return cls(**vars(parser.parse_args()))

    def __post_init__(self):
        assert self.input_events.exists(), f"{self.input_events} does not exist"
        assert self.input_video.exists(), f"{self.input_video} does not exist"


@dataclasses.dataclass
class EventsData:
    array: np.ndarray
    width: int
    height: int

    @classmethod
    def from_path(cls, path: pathlib.Path | str) -> EventsData:
        with open(path, "rb") as f:
            events = np.fromfile(f, dtype=np.uint8).reshape(-1, 8)

        meta_entry = events[-1]
        events = events[:-1]

        width = meta_entry[:2].view(dtype=np.uint16)[0]
        height = meta_entry[2:4].view(dtype=np.uint16)[0]
        n_events = meta_entry[4:].view(dtype=np.uint32)[0]
        assert n_events == len(events), f"Expected {n_events} events, got {len(events)}"

        ts_data = np.hstack(
            [events[:, :3], np.zeros((n_events, 1), dtype=np.uint8)]
        ).view(dtype=np.uint32)
        xs_data = events[:, 3:5].flatten().view(dtype=np.uint16).reshape(-1, 1)
        ys_data = events[:, 5:7].flatten().view(dtype=np.uint16).reshape(-1, 1)

        events = np.hstack(
            [ts_data, xs_data, ys_data, events[:, 7].reshape(-1, 1)]
        ).astype(np.float32)
        return cls(events, width, height)

    @property
    def num_events(self) -> int:
        return self.array.shape[0]


def read_video(path: pathlib.Path, verbose: bool = True) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iter_frames = range(num_frames)
    if verbose:
        iter_frames = tqdm.tqdm(iter_frames, total=num_frames, desc=f"Reading {path}")
    frames = []
    for _ in iter_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def crop_to_size(video: np.ndarray, width: int, height: int) -> np.ndarray:
    v_height, v_width = video.shape[1:3]
    assert v_width >= width, f"Video width {v_width} < {width}"
    assert v_height >= height, f"Video height {v_height} < {height}"

    target_prop = width / height
    assert target_prop > 1, "Width should be greater than height"
    target_width = int(target_prop * v_height)

    width_margin = int(v_width - target_width) // 2
    video = video[:, :, width_margin : width_margin + target_width]

    resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in video
    ]
    return np.array(resized)


def events_to_voxel_grid(events, num_bins, width, height):
    assert events.shape[1] == 4
    assert num_bins > 0
    assert width > 0
    assert height > 0

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + tis[valid_indices] * width * height,
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < num_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + (tis[valid_indices] + 1) * width * height,
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


class EventWindowIterator:
    def __init__(
        self,
        events: np.ndarray,
        counts: np.ndarray,
        window_length: int,
        stride: int = 1,
        offset: int = 0,
    ) -> None:
        self.events = events
        self.counts = counts
        self.event_index = 0
        self.count_index = 0
        self.window_length = window_length
        self.stride = stride
        self.offset = offset

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        window_start = self.offset + self.count_index

        if window_start >= self.counts.shape[0]:
            raise StopIteration

        window_end = window_start + self.window_length
        total_counts = self.counts[window_start:window_end].sum()
        window = self.events[self.event_index : self.event_index + total_counts]
        stride_counts = self.counts[window_start : window_start + self.stride].sum()
        self.event_index += stride_counts
        self.count_index += self.stride
        return window

    def __len__(self) -> int:
        res, rem = divmod(self.counts.shape[0] - self.offset, self.stride)
        return res + bool(rem)


@dataclasses.dataclass
class AccuracyStats:
    true_positives: list[int] = dataclasses.field(default_factory=list)
    true_negatives: list[int] = dataclasses.field(default_factory=list)
    false_positives: list[int] = dataclasses.field(default_factory=list)
    false_negatives: list[int] = dataclasses.field(default_factory=list)

    def update(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        tp = np.sum(predicted & actual)
        tn = np.sum(~predicted & ~actual)
        fp = np.sum(predicted & ~actual)
        fn = np.sum(~predicted & actual)
        self.true_positives.append(tp)
        self.true_negatives.append(tn)
        self.false_positives.append(fp)
        self.false_negatives.append(fn)

    def plot(self, ax: plt.Axes) -> None:
        ax.plot(self.true_positives, label="True Positives")
        ax.plot(self.false_positives, label="False Positives")
        ax.plot(self.true_negatives, label="True Negatives")
        ax.plot(self.false_negatives, label="False Negatives")
        ax.legend()


def to_exportable_frame(event_frame: np.ndarray, edged_frame: np.ndarray) -> np.ndarray:
    tp = (event_frame > 0) & (edged_frame > 0)
    fp = (event_frame > 0) & (edged_frame == 0)
    fn = (event_frame == 0) & (edged_frame > 0)
    tn = (event_frame == 0) & (edged_frame == 0)

    event_bgr = cv2.cvtColor(event_frame, cv2.COLOR_GRAY2BGR)
    event_bgr[tp] = [0, 255, 0]
    event_bgr[fp] = [0, 0, 255]
    event_bgr[fn] = [255, 0, 0]
    event_bgr[tn] = [255, 255, 255]
    return event_bgr


class Measurement:
    def __init__(self, reference_img: np.ndarray):
        self.reference_img = reference_img

    def measure(self, img: np.ndarray) -> float:
        raise NotImplementedError


class SIFTMeasurement(Measurement):
    def __init__(self, reference_img: np.ndarray):
        super().__init__(reference_img)
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(reference_img, None)

    def measure(self, img: np.ndarray) -> float:
        kp2, des2 = self.sift.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        return len(good)


class SSIMMeasurement(Measurement):
    def __init__(self, reference_img: np.ndarray):
        super().__init__(reference_img)

    def measure(self, img: np.ndarray) -> float:
        return skimage.measure.structural_similarity(
            self.reference_img, img, multichannel=False
        )


class PSNRMeasurement(Measurement):
    def __init__(self, reference_img: np.ndarray):
        super().__init__(reference_img)

    def measure(self, img: np.ndarray) -> float:
        return skimage.measure.peak_signal_noise_ratio(
            self.reference_img, img, multichannel=False
        )


def match_events_with_frame(
    events: np.ndarray,
    ts_counts: np.ndarray,
    reference_frame: np.ndarray,
    width: int,
    height: int,
    window_length: int,
    model: torch.nn.Module,
    measures: dict[str, type[Measurement]],
    kernel=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]], np.uint8),
    verbose: bool = True,
) -> tuple[AccuracyStats, dict[str, list[float]]]:
    stats = AccuracyStats()
    event_it = EventWindowIterator(events, ts_counts, window_length, stride=1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "experiments/videos/debug-match.mp4", fourcc, 20.0, (width, height)
    )

    ref_frame_gs = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    ref_frame_edged = (cv2.Canny(ref_frame_gs, 300, 400) > 0).astype(np.uint8) * 255
    ref_frame_edged = cv2.dilate(ref_frame_edged, np.ones((3, 3), np.uint8))
    measures = {name: measure(reference_frame) for name, measure in measures.items()}
    measure_results = {name: [] for name in measures}
    if verbose:
        event_it = tqdm.tqdm(event_it, total=len(event_it), desc="Matching events")
    for window in event_it:
        window = window.astype(int)
        frame = np.zeros((height, width), np.uint8)
        vg = events_to_voxel_grid(window, 5, width, height)
        vg = torch.from_numpy(vg).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            pred, _ = model(vg, None)
            pred = pred.squeeze().cpu().numpy()
            pred *= 255
            pred = pred.astype(np.uint8)
        for m_name, vals in measure_results.items():
            vals.append(measures[m_name].measure(pred))
        out.write(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR))

        stats.update(frame.ravel() > 0, ref_frame_edged.ravel() > 0)
    out.release()
    return stats, measure_results


if __name__ == "__main__":
    args = Args.from_cli()
    model = load_model("./pretrained/E2VID_lightweight.pth.tar")
    logging.info(f"Loading events from {args.input_events}")
    events = EventsData.from_path(args.input_events)
    first_timestamp = events.array[0, 0]
    logging.info(f"First timestamp: {first_timestamp}")
    logging.info(f"Width: {events.width}, Height: {events.height}")
    logging.info(f"Number of events: {len(events.array)}")
    video = read_video(args.input_video)
    logging.info(f"Resizing video to {events.width}x{events.height}")
    video = crop_to_size(video, events.width, events.height)

    _, ts_counts = np.unique(events.array[:, 0], return_counts=True)

    logging.info("Matching events with frame")
    checked_time_ms = 3000
    checked_counts = ts_counts[:checked_time_ms]
    checked_events = events.array[: checked_counts.sum()]

    stats, measures = match_events_with_frame(
        checked_events,
        checked_counts,
        video[0],
        events.width,
        events.height,
        window_length=50,
        model=model,
        measures={
            "SIFT": SIFTMeasurement,
            "SSIM": SSIMMeasurement,
            "PSNR": PSNRMeasurement,
        },
    )
    with open("experiments/measures.pkl", "wb") as f:
        pickle.dump(measures, f)

    fig, ax = plt.subplots()
    stats.plot(ax)
    plt.show()