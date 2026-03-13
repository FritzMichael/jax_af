from abc import ABC, abstractmethod
import os
from pathlib import Path

import h5py
import numpy as np


def open_data_reader(file_path: str | None = None) -> "DataReader":
    """Return the appropriate DataReader subclass based on file extension.

    .h5 / .hdf5  ->  H5Reader
    .npz         ->  NpzReader
    """
    if file_path is None:
        file_path = os.getenv("SAR_DATA_PATH")
    if not file_path:
        raise ValueError(
            "No data file path provided. Pass file_path explicitly or set SAR_DATA_PATH in the environment."
        )

    suffix = Path(file_path).suffix.lower()
    if suffix in (".h5", ".hdf5"):
        return H5Reader(file_path)
    elif suffix == ".npz":
        return NpzReader(file_path)
    raise ValueError(f"Unsupported file extension '{suffix}'. Expected .h5, .hdf5, or .npz.")


class DataReader(ABC):
    """Abstract base class for SAR data readers.

    Use open_data_reader(file_path) to get the appropriate reader based on
    file extension (.h5/.hdf5 -> H5Reader, .npz -> NpzReader).
    """

    @abstractmethod
    def read_range_profiles(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Read complex-valued range profiles for samples [start_idx, end_idx)."""

    @abstractmethod
    def read_positions(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Read 3-D radar positions (float32) for samples [start_idx, end_idx)."""

    @abstractmethod
    def read_velocities(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Read radar velocities (float32) for samples [start_idx, end_idx)."""

    @abstractmethod
    def get_slow_time_extent(self) -> int:
        """Return total number of slow-time samples."""


class H5Reader(DataReader):
    def __init__(self, file_path: str):
        self.archive = h5py.File(file_path, "r")
        self.range_profile_dict = dict(self.archive["meas_data"]["range_profile_ext"].attrs)
        self.radar_dict = dict(self.archive["meas_data"].attrs)

    def read_range_profiles(self, start_idx: int, end_idx: int) -> np.ndarray:
        return np.array(
            self.archive["meas_data"]["range_profile_ext"][start_idx:end_idx],
            dtype=np.complex64,
        ).squeeze()

    def read_positions(self, start_idx: int, end_idx: int) -> np.ndarray:
        pos_array =  np.array(
            self.archive["meas_data"]["posn_m2"][start_idx:end_idx],
            dtype=np.float32,
        ).squeeze()
        pos_array[:, 2] *= -1  # Flip z-axis to match right-handed coordinate system
        pos_array = pos_array[:, [1, 0, 2]]  # Reorder to (x, z, y)
        return pos_array

    def read_velocities(self, start_idx: int, end_idx: int) -> np.ndarray:
        vels =  np.array(
            self.archive["meas_data"]["vel_m2"][start_idx:end_idx],
            dtype=np.float32,
        ).squeeze()
        vels[:, 2] *= -1  # Flip z-axis to match right-handed coordinate system
        vels = vels[:, [1, 0, 2]]  # Reorder to (x, z, y)
        return vels

    def get_slow_time_extent(self) -> int:
        return self.archive["meas_data"]["range_profile_ext"].shape[0]


class NpzReader(DataReader):
    def __init__(self, file_path: str):
        self._data = np.load(file_path)

    def read_range_profiles(self, start_idx: int, end_idx: int) -> np.ndarray:
        return np.array(self._data["range_profiles"][start_idx:end_idx], dtype=np.complex64).squeeze()

    def read_positions(self, start_idx: int, end_idx: int) -> np.ndarray:
        return np.array(self._data["radar_positions"][start_idx:end_idx], dtype=np.float32).squeeze()

    def read_velocities(self, start_idx: int, end_idx: int) -> np.ndarray:
        return np.array(self._data["radar_velocities"][start_idx:end_idx], dtype=np.float32).squeeze()

    def get_slow_time_extent(self) -> int:
        return self._data["range_profiles"].shape[0]

    @property
    def image_center(self) -> np.ndarray:
        return self._data["image_center"]

    @property
    def sar_image(self) -> np.ndarray:
        return self._data["sar_image"]
