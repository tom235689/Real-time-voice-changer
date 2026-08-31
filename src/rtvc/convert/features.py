"""Mel spectrogram and pitch decoding, matched bit-for-bit against upstream RVC.

Reference implementations these were checked against:
    third_party/rvc/rmvpe.py     -- mel filterbank, cent decoding
    third_party/rvc/pipeline.py  -- coarse pitch bins, 2x feature upsampling

RVC computes these with librosa and torch. Reimplementing them on plain NumPy keeps
librosa and torch out of the runtime, where they would cost import time and memory for
no benefit. tests/test_preprocess.py pins the result against the original libraries.
"""

from __future__ import annotations

import numpy as np

from ..constants import ENC_SR

MEL_HOP = 160  # 16000 / 160 = 100 Hz, the generator's frame rate
MEL_NFFT = 1024
MEL_WIN = 1024
MEL_BINS = 128
MEL_FMIN = 30.0
MEL_FMAX = 8000.0
MEL_CLAMP = 1e-5

F0_MIN = 50.0
F0_MAX = 1100.0
RMVPE_THRESHOLD = 0.03


def _hz_to_mel_htk(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """Equivalent to librosa.filters.mel(htk=True, norm="slaney"), which is what RVC uses."""
    fft_hz = np.fft.rfftfreq(MEL_NFFT, 1.0 / ENC_SR)
    points = np.linspace(_hz_to_mel_htk(MEL_FMIN), _hz_to_mel_htk(MEL_FMAX), MEL_BINS + 2)
    hz = _mel_to_hz_htk(points)
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_hz[None, :]
    weights = np.zeros((MEL_BINS, fft_hz.size), dtype=np.float64)
    for i in range(MEL_BINS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz[2 : MEL_BINS + 2] - hz[:MEL_BINS])  # slaney area normalisation
    return (weights * enorm[:, None]).astype(np.float32)


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    """Magnitude spectrum matching torch.stft(center=True, pad_mode="reflect", periodic hann)."""
    window = np.hanning(MEL_WIN + 1)[:MEL_WIN].astype(np.float32)  # periodic, not symmetric
    pad = MEL_NFFT // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    n = 1 + x.shape[0] // MEL_HOP
    idx = np.arange(MEL_NFFT)[None, :] + MEL_HOP * np.arange(n)[:, None]
    spec = np.fft.rfft(padded[idx] * window, n=MEL_NFFT, axis=1)
    return np.abs(spec).T.astype(np.float32)  # (freq, frames)


class MelExtractor:
    def __init__(self) -> None:
        self.basis = mel_filterbank()

    def __call__(self, wav16: np.ndarray) -> np.ndarray:
        mel = self.basis @ stft_magnitude(wav16)
        return np.log(np.clip(mel, MEL_CLAMP, None)).astype(np.float32)  # (128, T)


# RMVPE emits 360 salience bins spaced 20 cents apart, starting at the cent value of
# 32.70 Hz. The 4-wide zero padding lets the 9-tap weighted average run at the edges.
_CENTS = np.pad(20.0 * np.arange(360) + 1997.3794084376191, (4, 4))


def decode_f0(hidden: np.ndarray, threshold: float = RMVPE_THRESHOLD) -> np.ndarray:
    """RMVPE salience (T, 360) -> f0 in Hz.

    Takes the salience-weighted mean of the 9 bins around the peak, matching RVC's
    to_local_average_cents. Frames whose peak salience is below the threshold are
    marked unvoiced.
    """
    center = np.argmax(hidden, axis=1) + 4
    salience = np.pad(hidden, ((0, 0), (4, 4)))
    rows = np.arange(salience.shape[0])[:, None]
    cols = center[:, None] + np.arange(-4, 5)[None, :]
    local_salience = salience[rows, cols]
    local_cents = _CENTS[cols]
    cents = (local_salience * local_cents).sum(1) / np.maximum(local_salience.sum(1), 1e-12)
    cents[np.max(salience, axis=1) <= threshold] = 0.0
    f0 = 10.0 * (2.0 ** (cents / 1200.0))
    f0[f0 == 10.0] = 0.0  # cents == 0 means unvoiced, not 10 Hz
    return f0.astype(np.float32)


_F0_MEL_MIN = 1127.0 * np.log(1 + F0_MIN / 700.0)
_F0_MEL_MAX = 1127.0 * np.log(1 + F0_MAX / 700.0)


def coarse_pitch(f0: np.ndarray) -> np.ndarray:
    """Continuous f0 -> integer bins 1..255, matching RVC's pipeline.py."""
    mel = 1127.0 * np.log(1.0 + f0.astype(np.float64) / 700.0)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - _F0_MEL_MIN) * 254.0 / (_F0_MEL_MAX - _F0_MEL_MIN) + 1.0
    mel[mel <= 1] = 1.0
    mel[mel > 255] = 255.0
    return np.rint(mel).astype(np.int64)
