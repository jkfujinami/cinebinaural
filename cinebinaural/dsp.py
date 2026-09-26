"""
cinebinaural DSP プリミティブ。フィルタはすべて SOS 形式(数値安定性) [調査 第II部]。
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, oaconvolve
from scipy.ndimage import minimum_filter1d, uniform_filter1d


# ---------- Butterworth (SOS) ----------
def _butter(kind, cutoff, sr, order):
    return butter(order, np.asarray(cutoff) / (sr / 2.0), btype=kind, output="sos")


def lowpass(x, cutoff, sr, order=4, zerophase=True):
    sos = _butter("low", cutoff, sr, order)
    return sosfiltfilt(sos, x, axis=-1) if zerophase else sosfilt(sos, x, axis=-1)


def highpass(x, cutoff, sr, order=4, zerophase=True):
    sos = _butter("high", cutoff, sr, order)
    return sosfiltfilt(sos, x, axis=-1) if zerophase else sosfilt(sos, x, axis=-1)


def bandpass(x, lo, hi, sr, order=4, zerophase=True):
    sos = _butter("band", [lo, hi], sr, order)
    return sosfiltfilt(sos, x, axis=-1) if zerophase else sosfilt(sos, x, axis=-1)


def crossover_zero_phase(x, fc, sr):
    """
    ゼロ位相の 2 分割。低域 = Butterworth 2 次を往復掛け(振幅は LR4 と同じ -6dB@fc)、
    高域 = x - 低域。Butterworth は電力相補なので高域も LR4 と同じ振幅になり、足すと元に完全に戻る。
    """
    low = sosfiltfilt(_butter("low", fc, sr, 2), x, axis=-1)
    return low, x - low


# ---------- RBJ biquad (peak / shelf) ----------
def rbj_sos(kind, fc, gain_db, q, sr):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        s = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
        a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
    elif kind == "highshelf":
        s = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s)]
        a = [(A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def high_shelf(x, fc, gain_db, sr, q=1 / np.sqrt(2)):
    return sosfilt(rbj_sos("highshelf", fc, gain_db, q, sr), x, axis=-1)


# ---------- 任意: missing fundamental [US5668885A] ----------
def virtual_bass(mono, sr, crossover, harm_lo, harm_hi, gain):
    """基音帯 LPF -> 全波整流(偶数次) -> 2 次高調波帯 BP -> ゲイン。加算用の倍音成分を返す。"""
    low = lowpass(mono, crossover, sr)
    rect = np.abs(low)
    harm = bandpass(rect - np.mean(rect), harm_lo, harm_hi, sr)
    scale = np.sqrt(np.mean(low ** 2)) / (np.sqrt(np.mean(harm ** 2)) + 1e-12)
    return harm * scale * gain


# ---------- ラウドネス / 最終リミッタ ----------
def integrated_lufs(stereo, sr):
    import pyloudnorm as pyln
    return pyln.Meter(sr).integrated_loudness(np.asarray(stereo).T)


def lookahead_limit(stereo, sr, ceiling_dbfs=-1.0, window_ms=15.0):
    """
    ゼロ位相ルックアヘッド・ピークリミッタ(オフライン)。
    必要ゲイン g[n] に幅 w の最小値フィルタ -> 同幅の移動平均。どのサンプルでも
    平滑化後ゲイン <= 必要ゲインが保証される。安全装置であり、通常はほぼ作動しない。
    戻り値: (処理後, 最大ゲインリダクション[dB])
    """
    ceil = 10 ** (ceiling_dbfs / 20.0)
    peak = np.max(np.abs(stereo), axis=0)
    g = np.minimum(1.0, ceil / np.maximum(peak, 1e-12))
    if np.all(g >= 1.0):
        return stereo, 0.0
    w = 2 * max(1, int(sr * window_ms / 1000.0)) + 1
    g = uniform_filter1d(minimum_filter1d(g, size=w, mode="nearest"), size=w, mode="nearest")
    y = stereo * g[None, :]
    p = np.max(np.abs(y))
    if p > ceil:            # 丸め誤差の保険
        y *= ceil / p
    return y, float(-20 * np.log10(np.min(g)))


def xcurve(x, sr, table, hinge_hz=2000.0, n=8192):
    """
    X カーブ(SMPTE ST 202 表1 の高域部分)を最小位相 FIR で掛ける。hinge_hz で曲がり始めを後ろにずらせる
    (サラウンド: 4kHz)。2kHz 以下(またはヒンジ以下)は 0dB。
    """
    f = np.fft.rfftfreq(n, 1 / sr)
    fx = np.array([t[0] for t in table], dtype=float) * (hinge_hz / table[0][0])
    fy = np.array([t[1] for t in table], dtype=float)
    g_db = np.interp(np.log(np.maximum(f, 1.0)), np.log(fx), fy, left=0.0, right=fy[-1])
    c = np.fft.irfft(np.log(10 ** (g_db / 20)), n=n)          # 実ケプストラム -> 最小位相
    w = np.zeros(n); w[0] = 1.0; w[1:n // 2] = 2.0; w[n // 2] = 1.0
    h = np.fft.irfft(np.exp(np.fft.rfft(c * w, n=n)), n=n)[: n // 2]
    return oaconvolve(x, h)[: len(x)]
