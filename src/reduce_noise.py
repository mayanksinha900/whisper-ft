import numpy as np
from librosa import stft, istft
from scipy.signal import filtfilt, fftconvolve
from typing import Optional

def sigmoid(x, shift, mult):
    return 1 / (1 + np.exp(-(x + shift) * mult))


def get_time_smoothed_representation(
        spectral, samplerate, 
        hop_length, time_constant_s=0.001):

    t_frames = time_constant_s * samplerate / float(hop_length)
    b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)

    return filtfilt([b], [1, b - 1], spectral, axis=-1, padtype=None)


def _smoothing_filter(n_grad_freq, n_grad_time):

    smoothing_filter = np.outer(
        np.concatenate([
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False), 
            np.linspace(1, 0, n_grad_freq + 2)
        ])[1:-1],
        np.concatenate([
            np.linspace(0, 1, n_grad_time + 1, endpoint=False), 
            np.linspace(1, 0, n_grad_time + 2)
        ])[1:-1]
    )

    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

    return smoothing_filter

class ReduceNoise:
    def __init__(
        self,
        sr = 8000, 
        prop_decrease = 1.0, 
        chunk_size = 600000, 
        padding = 30000, 
        n_fft = 1024, 
        time_constant_s = 2.0, 
        freq_mask_smooth_hz  = 1000, 
        time_mask_smooth_ms = 20
        ):
        
        self.sr = sr

        self._chunk_size = chunk_size
        self.padding = padding
        self._n_fft = n_fft

        self._win_length = self._n_fft
        self._hop_length = self._win_length // 4
        
        self._time_constant_s = time_constant_s
        self._prop_decrease = prop_decrease
        self._thresh_n_mult_nonstationary = 2
        self._sigmoid_slope_nonstationary = 10

        self._generate_mask_smoothing_filter(freq_mask_smooth_hz, time_mask_smooth_ms)

    def _generate_mask_smoothing_filter(self, 
                                        freq_mask_smooth_hz: int, 
                                        time_mask_smooth_ms: int):

        n_grad_freq = int(freq_mask_smooth_hz / (self.sr / (self._n_fft / 2)))
        n_grad_time = int(time_mask_smooth_ms / ((self._hop_length / self.sr) * 1000))

        self._smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    def _read_chunk(self, i1, i2):
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > self.n_frames:
            i2b = self.n_frames
        else:
            i2b = i2
        chunk = np.zeros((self.n_channels, i2 - i1))
        chunk[:, i1b - i1 : i2b - i1] = self.y[:, i1b:i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame) -> np.ndarray:

        i1 = start_frame - self.padding
        i2 = end_frame + self.padding
        padded_chunk = self._read_chunk(i1, i2)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1 : end_frame - i1]

    def _get_filtered_chunk(self, ind):

        start0 = ind * self._chunk_size
        end0 = (ind + 1) * self._chunk_size
        return self.filter_chunk(start_frame=start0, end_frame=end0)

    def _iterate_chunk(self, filtered_chunk, pos, end0, start0, ich):
        filtered_chunk0 = self._get_filtered_chunk(ich)
        filtered_chunk[:, pos : pos + end0 - start0] = filtered_chunk0[:, start0:end0]
        pos += end0 - start0

    def get_traces(self, y: list, 
                   start_frame: Optional[int] = None, 
                   end_frame: Optional[int] = None) -> np.ndarray:

        y = np.array(y)
        self.y = np.expand_dims(y, 0)
        self.n_channels, self.n_frames = self.y.shape

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.n_frames

        filtered_chunk = self.filter_chunk(start_frame=start_frame, end_frame=end_frame)

        return filtered_chunk.astype(np.float32).flatten()


    def spectral_gating_nonstationary(self, chunk: np.ndarray) -> np.ndarray:
        
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)
        for ci, channel in enumerate(chunk):

            sig_stft = stft(
                (channel), n_fft=self._n_fft, 
                hop_length=self._hop_length, win_length=self._win_length)
            abs_sig_stft = np.abs(sig_stft)
            sig_stft_smooth = get_time_smoothed_representation(
                abs_sig_stft, self.sr, self._hop_length, 
                time_constant_s=self._time_constant_s)
            sig_mult_above_thresh = (abs_sig_stft - sig_stft_smooth) / sig_stft_smooth
            sig_mask = sigmoid(
                sig_mult_above_thresh, -self._thresh_n_mult_nonstationary, 
                self._sigmoid_slope_nonstationary)
            sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")
            sig_mask = (sig_mask 
                        * self._prop_decrease 
                        + np.ones(np.shape(sig_mask)) 
                        * (1.0 - self._prop_decrease))
            sig_stft_denoised = sig_stft * sig_mask
            denoised_signal = istft(
                sig_stft_denoised, hop_length=self._hop_length, 
                win_length=self._win_length)
            denoised_channels[ci, : len(denoised_signal)] = denoised_signal

        return denoised_channels

    def _do_filter(self, chunk):

        chunk_filtered = self.spectral_gating_nonstationary(chunk)

        return chunk_filtered
