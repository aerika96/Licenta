import decimal
import numpy as np
import math
import logging


def round_half_up(number):
    result = decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)
    return result


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window +1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def frame_signal(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    signal_length = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if signal_length <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * signal_length- frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - signal_length,))
    padsignal = np.concatenate((sig, zeros))

    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframe_signal(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]

    assert np.shape(frames)[1] == frame_len, '"frames" matrix is  wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1))+ np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T

    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0:
        siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + win+ 1e-15
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction

    return rec_signal[0:siglen]


def magnitude_spectrum(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.')

    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    power_spec = 1.0 / NFFT * np.square(magnitude_spectrum(frames, NFFT))

    return power_spec


def logpowerspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    log_ps = 10 * np.log10(ps)

    if norm:
        return log_ps - np.max(log_ps)
    else:
        return log_ps


def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])
