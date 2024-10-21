from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import cv2
import numpy as np
import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

# homography matrix from the video to the events
V2E_HOMOGRAPHY = np.array(
    [
        [1.10537273e00, -1.71082704e-01, 5.68381444e01],
        [4.69460981e-02, 9.64188534e-01, -1.22149388e01],
        [1.47398028e-04, -4.38757896e-04, 1.00000000e00],
    ]
)


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
        xs_data = events[:, 3:5].view(dtype=np.uint16).reshape(-1, 1)
        ys_data = events[:, 5:7].view(dtype=np.uint16).reshape(-1, 1)

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
    """_summary_

    :param events: _description_
    :param num_bins: _description_
    :param width: _description_
    :param height: _description_
    :return: _description_
    """
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


def match_events_with_frame(
    events: np.ndarray,
    ts_counts: np.ndarray,
    edged_frame: np.ndarray,
    window_length: int = 50,
) -> AccuracyStats:
    stats = AccuracyStats()
    event_it = EventWindowIterator(events, ts_counts, window_length, stride=1)
    for window in event_it:
        ...

    return stats


if __name__ == "__main__":
    args = Args.from_cli()
    logging.info(f"Loading events from {args.input_events}")
    events = EventsData.from_path(args.input_events)
    first_timestamp = events.array[0, 0]
    logging.info(f"First timestamp: {first_timestamp}")
    logging.info(f"Width: {events.width}, Height: {events.height}")
    logging.info(f"Number of events: {len(events.array)}")
    logging.info(f"Input video: {args.input_video}")

    logging.info(f"Reading video from {args.input_video}")
    video = read_video(args.input_video)
    logging.info(f"Resizing video to {events.width}x{events.height}")
    video = crop_to_size(video, events.width, events.height)
