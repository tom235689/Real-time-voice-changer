"""Pin the preprocessing against the libraries RVC itself uses.

librosa and torch are test-only dependencies; the runtime path uses NumPy alone. A
mismatch here does not crash anything, it just makes the converted voice subtly wrong,
which is exactly the kind of bug that survives casual listening. So it is nailed down.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtvc.constants import ENC_SR
from rtvc.convert.features import (
    MEL_BINS,
    MEL_FMAX,
    MEL_FMIN,
    MEL_HOP,
    MEL_NFFT,
    MEL_WIN,
    coarse_pitch,
    decode_f0,
    mel_filterbank,
    stft_magnitude,
)


def test_mel_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(
        sr=ENC_SR, n_fft=MEL_NFFT, n_mels=MEL_BINS, fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True
    )
    got = mel_filterbank()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)


def test_stft_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(ENC_SR).astype(np.float32) * 0.1

    fft = torch.stft(
        torch.from_numpy(x),
        n_fft=MEL_NFFT,
        hop_length=MEL_HOP,
        win_length=MEL_WIN,
        window=torch.hann_window(MEL_WIN),
        center=True,
        return_complex=True,
    )
    ref = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2)).numpy()
    got = stft_magnitude(x)
    assert got.shape == ref.shape
    # Magnitudes span several orders of magnitude, so compare relatively.
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_stft_frame_rate_is_100hz():
    """Feature and pitch lengths only line up if the hop yields the generator frame rate."""
    for seconds in (0.5, 0.72, 1.0):
        x = np.zeros(int(ENC_SR * seconds), dtype=np.float32)
        assert stft_magnitude(x).shape[1] == 1 + int(ENC_SR * seconds) // MEL_HOP


def test_decode_f0_recovers_a_known_cent():
    hidden = np.zeros((3, 360), dtype=np.float32)
    hidden[:, 100] = 1.0
    cents = 20.0 * 100 + 1997.3794084376191
    np.testing.assert_allclose(decode_f0(hidden), 10.0 * 2.0 ** (cents / 1200.0), rtol=1e-4)


def test_decode_f0_treats_low_salience_as_unvoiced():
    hidden = np.full((4, 360), 0.001, dtype=np.float32)
    assert np.all(decode_f0(hidden) == 0.0)


def test_coarse_pitch_range_and_monotonicity():
    f0 = np.array([0.0, 50.0, 100.0, 220.0, 440.0, 1100.0, 2000.0], dtype=np.float32)
    bins = coarse_pitch(f0)
    assert bins.dtype == np.int64
    assert bins.min() >= 1 and bins.max() <= 255
    assert bins[0] == 1  # unvoiced
    assert np.all(np.diff(bins[1:]) >= 0)


def test_coarse_pitch_matches_rvc_formula():
    f0 = np.array([120.0, 300.0], dtype=np.float32)
    mel_min = 1127.0 * np.log(1 + 50.0 / 700.0)
    mel_max = 1127.0 * np.log(1 + 1100.0 / 700.0)
    m = 1127.0 * np.log(1 + f0.astype(np.float64) / 700.0)
    m = (m - mel_min) * 254.0 / (mel_max - mel_min) + 1.0
    np.testing.assert_array_equal(coarse_pitch(f0), np.rint(m).astype(np.int64))
