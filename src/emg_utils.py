"""
emg utilities for mindrove board: hardware interface, signal processing, and plotting.

consolidates all emg-related functionality:
  - mindrove wifi board interface (8 channels @ 500hz)
  - signal processing pipeline (notch + bandpass filters)
  - real-time plotting widget (pyqtgraph)
  - data buffering with timestamps
"""

import numpy as np
import time
from scipy import signal
from typing import Optional, Tuple

from mindrove import BoardShim, BoardIds, MindRoveInputParams
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore


# ============================================================
# configuration constants
# ============================================================

# hardware configuration
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD  # mindrove wifi board
NUM_CHANNELS = 8                          # 8 emg channels
SAMPLING_RATE = 500                       # 500 hz sampling
STREAM_BUFFER_SIZE = 450000               # internal ring buffer size

# signal processing parameters (from prosthetic/ml/util.py)
NOTCH_FREQ = 60.0                         # power line noise (hz)
NOTCH_Q = 30.0                            # notch filter quality factor
BANDPASS_LOW = 10.0                       # bandpass low cutoff (hz)
BANDPASS_HIGH = 200.0                     # bandpass high cutoff (hz)
BANDPASS_ORDER = 2                        # butterworth filter order

# plotting configuration
PLOT_WINDOW_SEC = 2.0                     # show last 2 seconds
PLOT_UPDATE_HZ = 30                       # 30 fps plotting


# ============================================================
# signal processing
# ============================================================

def filter_emg(data: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    """
    apply notch filter (60hz) and bandpass filter (10-200hz) to emg data.

    pipeline (from prosthetic/ml/util.py):
      1. notch @ 60hz (q=30): removes power line interference
      2. bandpass 10-200hz (butterworth order 2): isolates emg band
      3. zero-phase filtering (filtfilt): preserves phase information

    args:
      data: [num_samples, num_channels] raw emg
      fs: sampling rate (hz)

    returns:
      filtered: [num_samples, num_channels] filtered emg

    note: requires at least ~15 samples for filtfilt to work properly.
          for smaller batches, returns data unfiltered.
    """
    # filtfilt requires data length > padlen (typically ~3 * filter_order)
    # for 2nd order butterworth, padlen is ~9
    MIN_SAMPLES = 15

    if data.shape[0] < MIN_SAMPLES:
        # not enough samples to filter - return as-is
        # this typically only happens in first few batches
        return data.copy()

    # design notch filter
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)

    # design bandpass filter
    nyq = 0.5 * fs
    low = BANDPASS_LOW / nyq
    high = BANDPASS_HIGH / nyq
    b_band, a_band = signal.butter(BANDPASS_ORDER, [low, high], btype='band')

    # apply per channel (filtfilt for zero-phase)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        # apply notch filter
        channel_data = signal.filtfilt(b_notch, a_notch, channel_data)
        # apply bandpass filter
        channel_data = signal.filtfilt(b_band, a_band, channel_data)
        filtered[:, ch] = channel_data

    return filtered


# ============================================================
# mindrove hardware interface
# ============================================================

class MindroveInterface:
    """
    hardware interface to mindrove wifi board.

    handles:
      - board initialization and session management
      - data streaming at 500hz
      - channel selection (8 emg channels)
      - error handling and reconnection
    """

    def __init__(self):
        """initialize interface (lazy loading, call connect() to start)."""
        self.board_shim: Optional[BoardShim] = None
        self.is_connected = False
        self.sampling_rate = SAMPLING_RATE
        self.num_channels = NUM_CHANNELS
        self.emg_channel_indices = []  # board-specific channel indices

    def connect(self) -> bool:
        """
        initialize board and start streaming.

        returns:
          success: True if connected, False otherwise
        """
        if self.is_connected:
            print("[info] mindrove already connected")
            return True

        try:
            print("[info] connecting to mindrove wifi board...")

            # create board instance
            params = MindRoveInputParams()
            self.board_shim = BoardShim(BOARD_ID, params)

            # prepare session and start streaming
            self.board_shim.prepare_session()
            self.board_shim.start_stream(STREAM_BUFFER_SIZE)

            # allow stream to stabilize
            time.sleep(2.0)

            # verify stream is working
            data_count = self.board_shim.get_board_data_count()
            if data_count < 10:
                raise Exception(f"stream started but only {data_count} samples available")

            # get board info
            self.sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
            self.emg_channel_indices = BoardShim.get_emg_channels(BOARD_ID)

            # validate we have enough channels
            if len(self.emg_channel_indices) < NUM_CHANNELS:
                raise Exception(
                    f"board has {len(self.emg_channel_indices)} emg channels, "
                    f"need {NUM_CHANNELS}"
                )

            # use first NUM_CHANNELS
            self.emg_channel_indices = self.emg_channel_indices[:NUM_CHANNELS]

            print(f"[success] mindrove connected @ {self.sampling_rate}hz")
            print(f"  emg channels: {self.emg_channel_indices}")

            # flush initial buffer
            self.board_shim.get_board_data(self.board_shim.get_board_data_count())

            self.is_connected = True
            return True

        except Exception as e:
            print(f"[error] failed to connect to mindrove: {e}")
            self.disconnect()
            return False

    def get_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        retrieve recent emg data from board (non-blocking).

        args:
          num_samples: requested number of samples

        returns:
          timestamps: [actual_samples] seconds since start (float64)
          emg_data: [actual_samples, 8] raw emg voltages (float32)

        note: may return fewer samples than requested if not enough available
        """
        if not self.is_connected or not self.board_shim:
            return np.array([]), np.array([]).reshape(0, NUM_CHANNELS)

        try:
            # check available samples
            available = self.board_shim.get_board_data_count()
            if available == 0:
                return np.array([]), np.array([]).reshape(0, NUM_CHANNELS)

            # get what's available (up to num_samples)
            actual_samples = min(num_samples, available)
            raw_board_data = self.board_shim.get_board_data(actual_samples)

            # extract emg channels: raw_board_data is [all_channels, samples]
            emg_data = raw_board_data[self.emg_channel_indices, :].T  # → [samples, 8]

            # generate timestamps (assuming uniform sampling)
            # note: could use board timestamps if available
            timestamps = np.arange(emg_data.shape[0]) / self.sampling_rate

            return timestamps, emg_data.astype(np.float32)

        except Exception as e:
            print(f"[error] get_data failed: {e}")
            return np.array([]), np.array([]).reshape(0, NUM_CHANNELS)

    def disconnect(self):
        """stop streaming and release board."""
        if self.board_shim:
            try:
                if self.is_connected:
                    self.board_shim.stop_stream()
                    print("[info] mindrove stream stopped")
                self.board_shim.release_session()
                print("[info] mindrove session released")
            except Exception as e:
                print(f"[error] disconnect failed: {e}")
            finally:
                self.board_shim = None

        self.is_connected = False


# ============================================================
# real-time plotting widget
# ============================================================

class EMGPlotWidget(QtWidgets.QWidget):
    """
    real-time emg signal plotter using pyqtgraph (fast, 30fps+).

    displays:
      - 8 channels stacked vertically
      - last 2 seconds of data
      - auto-scaling per channel
      - channel labels
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # setup layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#111111')
        self.plot_widget.setLabel('left', 'emg amplitude')
        self.plot_widget.setLabel('bottom', 'time (seconds)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        # create plot curves for each channel
        self.curves = []
        colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24',
            '#6c5ce7', '#a29bfe', '#fd79a8', '#fdcb6e'
        ]

        for i in range(NUM_CHANNELS):
            curve = self.plot_widget.plot(
                pen=pg.mkPen(color=colors[i], width=1.5),
                name=f'ch{i+1}'
            )
            self.curves.append(curve)

        # data buffers (store last PLOT_WINDOW_SEC seconds)
        self.max_samples = int(SAMPLING_RATE * PLOT_WINDOW_SEC)
        self.time_buffer = np.array([])
        self.data_buffer = np.zeros((0, NUM_CHANNELS))
        self.time_offset = 0.0  # for continuous time axis

        # connection status indicator
        self.status_text = pg.TextItem(text="EMG: Connecting...", color=(255, 255, 0), anchor=(0, 0))
        self.plot_widget.addItem(self.status_text)
        self.status_text.setPos(0, 15)  # top-left position
        self.connected = False

    def update_plot(self, timestamps: np.ndarray, emg_data: np.ndarray, connected: bool = False):
        """
        update plot with new data.

        args:
          timestamps: [num_samples] relative timestamps (seconds)
          emg_data: [num_samples, 8] filtered emg data
          connected: whether emg hardware is connected
        """
        # update connection status indicator
        if connected != self.connected:
            self.connected = connected
            if connected:
                self.status_text.setText("EMG: Connected ✓")
                self.status_text.setColor((0, 255, 0))  # green
            else:
                self.status_text.setText("EMG: Not Connected")
                self.status_text.setColor((255, 0, 0))  # red

        if timestamps.size == 0 or emg_data.shape[0] == 0:
            return

        # append new data to buffers
        new_time = timestamps + self.time_offset
        self.time_buffer = np.concatenate([self.time_buffer, new_time])
        self.data_buffer = np.vstack([self.data_buffer, emg_data])

        # update time offset for next batch
        if timestamps.size > 0:
            self.time_offset = new_time[-1]

        # trim to last PLOT_WINDOW_SEC seconds
        if self.time_buffer.size > self.max_samples:
            keep = self.time_buffer.size - self.max_samples
            self.time_buffer = self.time_buffer[keep:]
            self.data_buffer = self.data_buffer[keep:, :]

        # update each channel curve
        for ch in range(NUM_CHANNELS):
            # offset each channel vertically for stacked display
            offset = (NUM_CHANNELS - ch - 1) * 2.0
            self.curves[ch].setData(
                self.time_buffer,
                self.data_buffer[:, ch] + offset
            )

    def clear(self):
        """clear all plot data."""
        self.time_buffer = np.array([])
        self.data_buffer = np.zeros((0, NUM_CHANNELS))
        self.time_offset = 0.0
        for curve in self.curves:
            curve.clear()
