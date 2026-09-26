"""
cinebinaural アップミックス: ステレオ -> 5.1(FL FR FC LFE SL SR)。理論書 Pillar 1。

  1. Demucs(htdemucs_ft)で 4 ステム(drums, bass, other, vocals)に分離。分離しきれなかった残差は other に足す
  2. ボーカルの中央成分(左右で同じに鳴っている成分)を FC へ
     - 左右の類似度 ψ = 2|L R*| / (|L|^2 + |R|^2) を STFT の各ビンで求める(Avendano & Jot 2004)
  3. 各ステムの、左右で相関の低い成分(響き・広がり)を SL/SR へ
     - 相互相関係数 φ = |E[L R*]| / sqrt(E|L|^2 E|R|^2) が低いほど響きとみなす(同上)
  4. 残り(直接音)を FL/FR へ。LFE は空(ベースマネジメントでメインの低域はサブへ行くため、二重になる)

  ITU ダウンミックス(L = FL + 0.707 FC + 0.707 SL)で元のステレオに戻るようにゲインを決める:
  FC = √2 × 中央成分、SL/SR = √2 × 響き。普通のステレオ再生ではバランスが元のまま保たれる。
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, lfilter

NPERSEG = 2048
HOP = 512
CORR_TIME_MS = 60.0          # 相関を求めるときの時間平均
CENTER_THRESHOLD = 0.8       # 類似度 ψ がこれ以上の成分をセンター寄りとみなす(1 で完全に中央)


def separate(x, sr, model_name="htdemucs_ft"):
    """Demucs でステム分離。x: (N, 2)。戻り {名前: (2, N)}。"""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = get_model(model_name)
    model.eval()
    wav = convert_audio(torch.tensor(x.T, dtype=torch.float32), sr, model.samplerate, model.audio_channels)
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    import time
    t0 = time.time()
    with torch.no_grad():
        try:
            out = apply_model(model, ((wav - mu) / sd)[None], device=device, split=True, overlap=0.25,
                              progress=True)[0]
        except Exception as e:
            print(f"  Demucs: {device} で失敗したので CPU で実行 ({type(e).__name__}: {e})")
            device = "cpu"
            out = apply_model(model, ((wav - mu) / sd)[None], device="cpu", split=True, overlap=0.25,
                              progress=True)[0]
    print(f"  Demucs: {model_name} を {device} で実行、{time.time() - t0:.1f} 秒")
    out = out * sd + mu
    stems = {}
    for name, s in zip(model.sources, out):
        y = convert_audio(s, model.samplerate, sr, 2).numpy()
        z = np.zeros((2, x.shape[0]))
        m = min(y.shape[1], x.shape[0])
        z[:, :m] = y[:, :m]
        stems[name] = z
    stems["other"] = stems["other"] + (x.T - sum(stems.values()))   # 残差を other へ(総和が元に一致)
    return stems


def _smooth(a, sr):
    """STFT のフレーム方向に 1 次の移動平均(時定数 CORR_TIME_MS)。"""
    coef = np.exp(-HOP / (sr * CORR_TIME_MS / 1000.0))
    return lfilter([1 - coef], [1.0, -coef], a, axis=-1)


def _stats(Z, sr):
    L, R = Z
    pll = _smooth(np.abs(L) ** 2, sr)
    prr = _smooth(np.abs(R) ** 2, sr)
    plr = _smooth(L * np.conj(R), sr)
    return pll, prr, plr


def _stft(x, sr):
    return stft(x, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary="even", padded=True)[2]


def _istft(Z, sr, n):
    y = istft(Z, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=True)[1]
    out = np.zeros((2, n))
    m = min(n, y.shape[-1])
    out[:, :m] = y[:, :m]
    return out


def split_center(x2, sr):
    """ステレオ (2, N) -> (中央成分 (N,), 残り (2, N))。残り + 中央 = 元(左右とも)。"""
    Z = _stft(x2, sr)
    pll, prr, plr = _stats(Z, sr)
    psi = 2 * np.abs(plr) / np.maximum(pll + prr, 1e-20)
    mask = np.clip((psi - CENTER_THRESHOLD) / (1 - CENTER_THRESHOLD), 0.0, 1.0)
    C = mask * (Z[0] + Z[1]) / 2
    c = _istft(np.stack([C, C]), sr, x2.shape[1])[0]
    return c, x2 - c[None, :]


def split_ambience(x2, sr):
    """ステレオ (2, N) -> (直接音 (2, N), 響き (2, N))。直接音 + 響き = 元。"""
    Z = _stft(x2, sr)
    pll, prr, plr = _stats(Z, sr)
    phi = np.abs(plr) / np.sqrt(np.maximum(pll * prr, 1e-40))
    mask = np.clip(1.0 - phi, 0.0, 1.0)
    A = _istft(Z * mask[None], sr, x2.shape[1])
    return x2 - A, A


def split_ambience_ls(x2, sr):
    """
    ステレオ (2, N) -> (直接音 (2, N), 響き (2, N))。最小二乗推定(Faller 2006)。
    各時間周波数ビンで L = S + N1, R = a S + N2(S は直接音、N1・N2 は互いに無相関で同じパワーの響き)とみなし、
    平滑化した共分散から P_N(共分散行列の小さい方の固有値)と P_S を求め、N1・N2 を L・R の線形結合で推定する
    (Wiener 解 w = Rxx^-1 E[x N*])。マスク方式と違い、響きの推定が L・R 両方を使うので直接音の漏れが小さい。
    直接音 + 響き = 元。
    """
    Z = _stft(x2, sr)
    pll, prr, plr = _stats(Z, sr)
    pn = np.maximum(0.5 * (pll + prr - np.sqrt((pll - prr) ** 2 + 4 * np.abs(plr) ** 2)), 0.0)
    det = np.maximum(pll * prr - np.abs(plr) ** 2, 1e-12 * (pll * prr + 1e-30))
    # Rxx = [[pll, plr], [plr*, prr]]、E[x N1*] = [pn, 0]、E[x N2*] = [0, pn]。N^ = w^H x
    # Rxx^-1 = [[prr, -plr], [-plr*, pll]] / det
    w1 = np.stack([prr * pn, -np.conj(plr) * pn]) / det           # N1 の重み(L, R への係数、共役前)
    w2 = np.stack([-plr * pn, pll * pn]) / det
    A = np.stack([np.conj(w1[0]) * Z[0] + np.conj(w1[1]) * Z[1],
                  np.conj(w2[0]) * Z[0] + np.conj(w2[1]) * Z[1]])
    a = _istft(A, sr, x2.shape[1])
    return x2 - a, a


def repan_center(x2, sr):
    """
    ステレオ (2, N) -> (中央成分 (N,), 残り (2, N))。すべての音源を L・C・R に振り直す(Avendano & Jot 2004 の考え方)。
    振幅パンの音源 L = a s, R = b s(a >= b)なら、共通部分 b s を中央へ、残り (a - b) s を L に残す
    (L 寄りの音は L と C の間に分配される)。ゲイン = 2 min(|L|,|R|) / (|L|+|R|) × コヒーレンス(響きは動かさない)。
    残り + 中央 = 元(左右とも)なので、ITU ダウンミックスで元に戻る。
    """
    Z = _stft(x2, sr)
    pll, prr, plr = _stats(Z, sr)
    al, ar = np.sqrt(pll), np.sqrt(prr)
    k = 2 * np.minimum(al, ar) / np.maximum(al + ar, 1e-20)
    coh = np.abs(plr) / np.maximum(al * ar, 1e-20)
    C = np.clip(k * coh, 0.0, 1.0) * (Z[0] + Z[1]) / 2
    c = _istft(np.stack([C, C]), sr, x2.shape[1])[0]
    return c, x2 - c[None, :]


def decorrelator(sr, seed, n=4096, f_lo=200.0, f_full=400.0, points_per_oct=3, f_top=20000.0):
    """
    振幅が平坦で位相だけがランダムな FIR(サラウンドの無相関化, Avendano & Jot 2004 / Faller 2006)。
    位相は 1/points_per_oct oct ごとのランダム値(±π)を補間して滑らかにし(群遅延を数 ms に抑える)、
    f_lo 未満は 0(低域は動かさない)、f_lo-f_full でなめらかに効かせる。f_top より上も 0 に戻す
    (ナイキストのビンは実数しか取れず、位相が残ると振幅が落ちる。実測 -1.1dB)。
    """
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1 / sr)
    nodes = f_lo * 2 ** (np.arange(0, np.log2(sr / 2 / f_lo) + 1 / points_per_oct, 1 / points_per_oct))
    ph = np.interp(np.log2(np.maximum(f, 1.0)), np.log2(nodes), rng.uniform(-np.pi, np.pi, len(nodes)))
    w = np.clip(np.log2(np.maximum(f, 1e-3) / f_lo) / np.log2(f_full / f_lo), 0.0, 1.0)
    w = w * np.clip((sr / 2 - f) / (sr / 2 - f_top), 0.0, 1.0)
    h = np.fft.irfft(np.exp(1j * ph * w), n=n)
    return np.fft.fftshift(h)


def _cached_stems(x, sr, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k].astype(np.float64) for k in d.files}
    stems = separate(x, sr)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, **{k: v.astype(np.float32) for k, v in stems.items()})
        return {k: v.astype(np.float32).astype(np.float64) for k, v in stems.items()}
    return stems


def upmix_stereo_to_5p1(in_wav, out_wav, verbose=True, center="vocals", ambience="mask",
                        surround_decorr=False, surround_delay_ms=0.0, stems_cache=None):
    """
    center: "vocals" = ボーカルの中央成分だけ FC へ(最初の版)/ "repan" = 全ステムを L・C・R に振り直す
    ambience: "mask" = 1 - コヒーレンスのマスク(最初の版)/ "ls" = 最小二乗推定(Faller 2006)
    surround_decorr: サラウンドを無相関化(SL・SR で別のランダム位相)
    surround_delay_ms: サラウンドの遅延(先行音効果で前方の定位を守る。Haas 1951: 5-30ms / Litovsky 1999: 打音は約 10ms)
    stems_cache: Demucs の分離結果を保存・再利用する .npz(バリアントを同じステムから作るため)
    """
    if center == "vocals" and ambience == "mask" and not surround_decorr and not surround_delay_ms and not stems_cache:
        return _upmix_v1(in_wav, out_wav, verbose)
    log = print if verbose else (lambda *a: None)
    x, sr = sf.read(in_wav, always_2d=True)
    if x.shape[1] != 2:
        raise ValueError(f"expected stereo, got {x.shape[1]}ch")
    n = x.shape[0]
    log(f"upmix: {in_wav} ({sr}Hz, {n / sr:.1f}s) -> {out_wav}  [center={center}, ambience={ambience}, "
        f"decorr={surround_decorr}, delay={surround_delay_ms:g}ms]")
    stems = _cached_stems(x, sr, stems_cache)
    amb_split = split_ambience_ls if ambience == "ls" else split_ambience
    front, amb, cen = np.zeros((2, n)), np.zeros((2, n)), np.zeros(n)
    for name in ("vocals", "drums", "bass", "other"):
        sig = stems[name]
        if center == "vocals" and name == "vocals":
            c, sig = split_center(sig, sr)
            cen += c
        d, a = amb_split(sig, sr)
        if center == "repan":
            c, d = repan_center(d, sr)
            cen += c
        front += d
        amb += a
        log(f"      {name:6}: 響きの割合 {10 * np.log10(np.mean(a ** 2) / max(np.mean(sig ** 2), 1e-20)):+6.1f} dB")
    k = np.sqrt(2.0)
    out = np.zeros((n, 6))
    out[:, 0], out[:, 1] = front[0], front[1]
    out[:, 2] = k * cen
    sl, sr_ = k * amb[0], k * amb[1]
    if surround_decorr:
        sl = oaconvolve_same(sl, decorrelator(sr, 11))
        sr_ = oaconvolve_same(sr_, decorrelator(sr, 23))
    if surround_delay_ms:
        dly = int(round(surround_delay_ms / 1000 * sr))
        sl = np.concatenate([np.zeros(dly), sl[: n - dly]])
        sr_ = np.concatenate([np.zeros(dly), sr_[: n - dly]])
    out[:, 4], out[:, 5] = sl, sr_
    L = out[:, 0] + out[:, 2] / k + out[:, 4] / k
    R = out[:, 1] + out[:, 2] / k + out[:, 5] / k
    err = 10 * np.log10(np.mean((np.stack([L, R], 1) - x) ** 2) / np.mean(x ** 2))
    peak = np.max(np.abs(out))
    log(f"  ITU ダウンミックスと元の差: {err:.1f} dB / 5.1 のピーク {peak:.2f}")
    for i, nm in enumerate(["FL", "FR", "FC", "LFE", "SL", "SR"]):
        log(f"      {nm:3}: {10 * np.log10(np.mean(out[:, i] ** 2) + 1e-20):6.1f} dB")
    sf.write(out_wav, out, sr, subtype="FLOAT" if peak > 1.0 else "PCM_24")
    return out_wav


def oaconvolve_same(x, h):
    """h の中心(群遅延 len(h)/2)を打ち消して、x と同じ長さ・同じ時刻に揃えた畳み込み。"""
    from scipy.signal import oaconvolve
    y = oaconvolve(x, h)
    s = len(h) // 2
    return y[s: s + len(x)]


def _upmix_v1(in_wav, out_wav, verbose=True):
    log = print if verbose else (lambda *a: None)
    x, sr = sf.read(in_wav, always_2d=True)
    if x.shape[1] != 2:
        raise ValueError(f"expected stereo, got {x.shape[1]}ch")
    n = x.shape[0]
    log(f"upmix: {in_wav} ({sr}Hz, {n / sr:.1f}s) -> {out_wav}")
    log("  [1] Demucs htdemucs_ft でステム分離")
    stems = separate(x, sr)
    for k, v in stems.items():
        log(f"      {k:6}: {10 * np.log10(np.mean(v ** 2) + 1e-20):6.1f} dB")

    log("  [2] ボーカルの中央成分 -> FC")
    center, vocal_rest = split_center(stems["vocals"], sr)

    log("  [3] 響き(左右で相関の低い成分)-> SL/SR、直接音 -> FL/FR")
    front = np.zeros((2, n))
    amb = np.zeros((2, n))
    for name, sig in (("vocals", vocal_rest), ("drums", stems["drums"]),
                      ("bass", stems["bass"]), ("other", stems["other"])):
        d, a = split_ambience(sig, sr)
        front += d
        amb += a
        log(f"      {name:6}: 響きの割合 {10 * np.log10(np.mean(a ** 2) / max(np.mean(sig ** 2), 1e-20)):+6.1f} dB")

    k = np.sqrt(2.0)
    out = np.zeros((n, 6))                       # FL FR FC LFE SL SR
    out[:, 0], out[:, 1] = front[0], front[1]
    out[:, 2] = k * center
    out[:, 4], out[:, 5] = k * amb[0], k * amb[1]

    # ITU ダウンミックスで元に戻るかの確認
    L = out[:, 0] + out[:, 2] / k + out[:, 4] / k
    R = out[:, 1] + out[:, 2] / k + out[:, 5] / k
    err = 10 * np.log10(np.mean((np.stack([L, R], 1) - x) ** 2) / np.mean(x ** 2))
    peak = np.max(np.abs(out))
    log(f"  ITU ダウンミックスと元の差: {err:.1f} dB / 5.1 のピーク {peak:.2f}")
    for i, name in enumerate(["FL", "FR", "FC", "LFE", "SL", "SR"]):
        log(f"      {name:3}: {10 * np.log10(np.mean(out[:, i] ** 2) + 1e-20):6.1f} dB")
    if peak > 1.0:
        log("  (ピークが 1.0 を超えるので float で保存)")
    sf.write(out_wav, out, sr, subtype="FLOAT" if peak > 1.0 else "PCM_24")
    return out_wav


__all__ = ["upmix_stereo_to_5p1", "separate", "split_center", "split_ambience", "split_ambience_ls",
           "repan_center", "decorrelator"]
