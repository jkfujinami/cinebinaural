"""
cinebinaural パイプライン: 5.1ch ロスレス・マスター -> IMAX バイノーラル 2ch。

  5.1 読込
   -> LFE: +10dB, LPF 120Hz(無指向で両耳へ) [任意: missing fundamental]
   -> 方向 5ch: 音色中立 BRIR(直接音 + image source 初期反射 + 拡散残響)で畳み込み
   -> 知覚的トーナル(4kHz -2dB シェルフ)
   -> [任意] ヘッドホン EQ
   -> LUFS 正規化 -> セーフティ・リミッタ
"""
import os

import numpy as np
import soundfile as sf

from . import config as C
from .config import PipelineConfig
from . import dsp
from .spatial import HRTF, load_hrtf, build_brirs, build_sub_brir, render_binaural, direct_arrival_samples


def _load_5p1(path, sr):
    x, file_sr = sf.read(path, always_2d=True)
    if file_sr != sr:
        raise ValueError(f"input SR {file_sr} != {sr}")
    if x.shape[1] != 6:
        raise ValueError(f"expected 6ch (5.1), got {x.shape[1]}ch")
    return {name: x[:, i].astype(np.float64) for i, name in enumerate(C.CH_ORDER_5P1)}


def _finalize(stereo, sr, target_lufs, allow_limiting=True):
    before = dsp.integrated_lufs(stereo, sr)
    gain_db = target_lufs - before
    if not allow_limiting:
        # アタックを丸めないよう、ピークが天井を超えない範囲に音量を抑える(LUFS は目標より下がることがある)
        peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-12)
        gain_db = min(gain_db, C.LIMIT_CEILING_DBFS - peak_db)
        return stereo * 10 ** (gain_db / 20), dsp.integrated_lufs(stereo * 10 ** (gain_db / 20), sr), 0.0
    stereo = stereo * 10 ** (gain_db / 20)
    stereo, gr_db = dsp.lookahead_limit(stereo, sr, C.LIMIT_CEILING_DBFS, C.LIMITER_WINDOW_MS)
    return stereo, dsp.integrated_lufs(stereo, sr), gr_db


def _write(path, stereo, sr):
    sf.write(path, stereo.T, sr, subtype="PCM_24")


def process_5p1(input_path, output_path, cfg: PipelineConfig = None, verbose=True):
    cfg = cfg or PipelineConfig()
    sr = cfg.sample_rate
    log = print if verbose else (lambda *a: None)

    log(f"cinebinaural: {input_path} -> {output_path}")
    ch = _load_5p1(input_path, sr)

    lfe = dsp.lowpass(ch["LFE"], C.LFE_LP_HZ, sr) * 10 ** (cfg.lfe_gain_db / 20)
    log(f"  LFE: +{cfg.lfe_gain_db:.0f}dB, LPF {C.LFE_LP_HZ:.0f}Hz, 無指向")
    if cfg.virtual_bass:
        lfe = lfe + dsp.virtual_bass(lfe, sr, C.VB_CROSSOVER_HZ, C.VB_HARMONIC_LO,
                                     C.VB_HARMONIC_HI, C.VB_GAIN)
        log("  virtual bass: ON")

    log(f"  BRIR: HRTF={cfg.hrtf} ({cfg.hrtf_path}), room={cfg.room_model}, "
        f"surround={'array' if cfg.surround_array else 'point'}, front=±{cfg.front_angle:g}°, "
        f"timbre-neutral={cfg.timbre_neutral_strength}")
    hrtf = load_hrtf(cfg, sr)
    brirs = build_brirs(hrtf, sr, cfg, log=log)
    mains = {k: ch[k] for k in C.SCREEN_CHANNELS + C.SURROUND_CHANNELS}
    if cfg.tonal == "xcurve":
        # 劇場では X カーブを各スピーカー系統に掛ける(SMPTE ST 202)。スクリーンは 2kHz、サラウンドは 4kHz から。
        for k in mains:
            hinge = C.XCURVE_SURROUND_HINGE if k in C.SURROUND_CHANNELS else 2000.0
            mains[k] = dsp.xcurve(mains[k], sr, C.XCURVE_ST202, hinge_hz=hinge)
        log("  tonal: X カーブ (SMPTE ST 202 表1、サラウンドはヒンジ 4kHz)")
    if cfg.surround_gain_db:
        # チャンネルレベル(ベースマネジメントより前): サブに回る低音も同じ比率で下がる
        for k in C.SURROUND_CHANNELS:
            mains[k] = mains[k] * 10 ** (cfg.surround_gain_db / 20)
        log(f"  surround level: {cfg.surround_gain_db:+g}dB")
    if cfg.bass_management:
        # IMAX のサブベース: 5ch の低域 + LFE をスクリーン中央下のサブから鳴らす。サブも同じ部屋を通す。
        # サブの BRIR は他の ch と同じく直接音が t=0 に校正されているので、時間合わせは不要。
        sub = lfe
        for k in list(mains):
            low, mains[k] = dsp.crossover_zero_phase(mains[k], cfg.bass_crossover_hz, sr)
            sub = sub + low
        brirs = {**brirs, "SUB": build_sub_brir(hrtf, sr, cfg, log=log)}
        stereo = render_binaural({**mains, "SUB": sub}, None, brirs)
        log(f"  bass management: {cfg.bass_crossover_hz:g}Hz (ゼロ位相 LR4 相当), 5ch 低域 + LFE -> SUB")
    else:
        if cfg.lfe_align:
            # 劇場ではサブウーファーとメインの到達を席で揃える。LFE をスクリーン ch の直接音に合わせて遅らせる。
            delay = direct_arrival_samples(brirs)
            lfe = np.concatenate([np.zeros(delay), lfe])
            log(f"  LFE align: +{delay / sr * 1000:.2f}ms (スクリーン ch の直接音の到達時刻)")
        stereo = render_binaural(mains, lfe, brirs)

    if cfg.tonal in C.TONAL_PRESETS:
        fc, gain, q = C.TONAL_PRESETS[cfg.tonal]
        stereo = dsp.high_shelf(stereo, fc, gain, sr, q=q)
        log(f"  tonal: {cfg.tonal} ({gain:+g}dB shelf @ {fc:g}Hz, Q {q:g})")
    if cfg.headphone_eq:
        from .headphone_eq import apply_autoeq
        stereo = apply_autoeq(stereo, sr, cfg.headphone_eq)
        log(f"  headphone EQ: {cfg.headphone_eq}")

    stereo, lufs, gr = _finalize(stereo, sr, cfg.target_lufs, cfg.allow_limiting)
    _write(output_path, stereo, sr)
    lim = f"limiter GR max {gr:.1f}dB" if cfg.allow_limiting else "limiter 不使用"
    log(f"  out: {lufs:.1f} LUFS, peak {20 * np.log10(np.max(np.abs(stereo)) + 1e-12):.1f}dBFS, {lim}, "
        f"{stereo.shape[1] / sr:.1f}s")
    return output_path


def _read_channel(path, idx, sr, block_s=60):
    """6ch ファイルから 1 チャンネルだけ全長で読む。値は sf.read(path)[:, idx] と同じ(変換は 1 サンプルごと)。"""
    with sf.SoundFile(path) as f:
        if f.samplerate != sr or f.channels != 6:
            raise ValueError(f"expected 6ch {sr}Hz, got {f.channels}ch {f.samplerate}Hz")
        out = np.empty(f.frames, dtype=np.float64)
        pos, blk = 0, int(block_s * sr)
        while pos < f.frames:
            x = f.read(min(blk, f.frames - pos), dtype="float64", always_2d=True)
            out[pos:pos + len(x)] = x[:, idx]
            pos += len(x)
    return out


def process_5p1_lowmem(input_path, output_path, cfg: PipelineConfig = None, work_dir=None, verbose=True):
    """
    process_5p1 と同じ関数を、同じ順序で、同じ全長の配列に対して呼ぶ(区切らない)。違いはメモリのやりくりだけ:
      - 入力を 1 チャンネルずつ読む
      - クロスオーバー後の各チャンネルは使うまで work_dir に退避(np.save / np.load はバイト列をそのまま保存)
      - 使い終わった配列はすぐ解放
      - 畳み込みの足し込み先・サブの信号・各チャンネルは work_dir 上のメモリマップ(値と演算は同じ)
    出力は process_5p1 とビット単位で同じになるように作ってあり、ハッシュで確認する。
    """
    from scipy.signal import oaconvolve
    cfg = cfg or PipelineConfig()
    sr = cfg.sample_rate
    log = print if verbose else (lambda *a: None)
    if cfg.virtual_bass:
        raise NotImplementedError("lowmem は virtual_bass 未対応")
    work_dir = work_dir or (os.path.splitext(output_path)[0] + "_work")
    os.makedirs(work_dir, exist_ok=True)
    idx = {name: i for i, name in enumerate(C.CH_ORDER_5P1)}
    log(f"cinebinaural (lowmem, 区切らない): {input_path} -> {output_path}  (退避先 {work_dir})")

    lfe = dsp.lowpass(_read_channel(input_path, idx["LFE"], sr), C.LFE_LP_HZ, sr) * 10 ** (cfg.lfe_gain_db / 20)
    hrtf = load_hrtf(cfg, sr)
    brirs = build_brirs(hrtf, sr, cfg, log=log)
    names = list(C.SCREEN_CHANNELS + C.SURROUND_CHANNELS)

    sub = lfe if cfg.bass_management else None
    for k in names:
        sig = _read_channel(input_path, idx[k], sr)
        if cfg.tonal == "xcurve":
            hinge = C.XCURVE_SURROUND_HINGE if k in C.SURROUND_CHANNELS else 2000.0
            sig = dsp.xcurve(sig, sr, C.XCURVE_ST202, hinge_hz=hinge)
        if cfg.surround_gain_db and k in C.SURROUND_CHANNELS:
            sig = sig * 10 ** (cfg.surround_gain_db / 20)
        if cfg.bass_management:
            low, sig = dsp.crossover_zero_phase(sig, cfg.bass_crossover_hz, sr)
            sub = sub + low
            del low
        np.save(os.path.join(work_dir, f"{k}.npy"), sig)
        log(f"  前処理 {k} -> 退避")
        del sig

    if cfg.bass_management:
        brirs = {**brirs, "SUB": build_sub_brir(hrtf, sr, cfg, log=log)}
        order, lfe_add = names + ["SUB"], None
        n_sub = len(sub)
        np.save(os.path.join(work_dir, "SUB.npy"), sub)       # 畳み込みの間はメモリに置かない
        del sub
        sub = None
    else:
        order = names
        lfe_add = lfe
        if cfg.lfe_align:
            delay = direct_arrival_samples(brirs)
            lfe_add = np.concatenate([np.zeros(delay), lfe])

    # render_binaural と同じ: out の長さ、足す順序(チャンネル順 -> 耳の順)
    n = n_sub if cfg.bass_management else len(lfe)
    m = max(b.shape[1] for b in brirs.values())
    # 足し込み先は work_dir 上のメモリマップ(0 で初期化、足し算の中身と順序は np.zeros のときと同じ)
    out = np.lib.format.open_memmap(os.path.join(work_dir, "out.npy"), mode="w+", dtype=np.float64,
                                    shape=(2, n + m - 1))
    out[...] = 0.0
    for name in order:
        sig = np.load(os.path.join(work_dir, f"{name}.npy"), mmap_mode="r")
        for ear in (0, 1):
            y = oaconvolve(sig, brirs[name][ear])
            out[ear, : len(y)] += y
            del y
        del sig
        log(f"  畳み込み {name}")
    if lfe_add is not None:
        out[:, : len(lfe_add)] += lfe_add[None, :]
    del sub, lfe, lfe_add
    stereo = out
    del out

    if cfg.tonal in C.TONAL_PRESETS:
        fc, gain, q = C.TONAL_PRESETS[cfg.tonal]
        stereo = dsp.high_shelf(stereo, fc, gain, sr, q=q)
    if cfg.headphone_eq:
        from .headphone_eq import apply_autoeq
        stereo = apply_autoeq(stereo, sr, cfg.headphone_eq)

    stereo, lufs, gr = _finalize(stereo, sr, cfg.target_lufs, cfg.allow_limiting)
    _write(output_path, stereo, sr)
    for k in names + (["SUB"] if cfg.bass_management else []) + ["out"]:
        os.remove(os.path.join(work_dir, f"{k}.npy"))
    lim = f"limiter GR max {gr:.1f}dB" if cfg.allow_limiting else "limiter 不使用"
    log(f"  out: {lufs:.1f} LUFS, {lim}, {stereo.shape[1] / sr:.1f}s")
    return output_path


class _StreamingLoudness:
    """
    ITU-R BS.1770 の積分ラウドネスを、区切って流し込みながら求める。pyloudnorm.Meter.integrated_loudness と
    同じ計算(K 特性フィルタは状態を引き継いだ lfilter、400ms ブロックは同じ区間・同じ順序の総和)にしてある。
    """

    def __init__(self, sr, n_total, channels=2):
        import pyloudnorm as pyln
        from scipy.signal import lfilter_zi  # noqa: F401  (状態は 0 から始める: pyloudnorm と同じ)
        self.sr, self.ch = sr, channels
        self.stages = [(f.b, f.a, f.passband_gain) for f in pyln.Meter(sr)._filters.values()]
        self.zi = [[np.zeros(max(len(b), len(a)) - 1) for _ in range(channels)] for b, a, _ in self.stages]
        self.T_g, self.step = 0.400, 0.25
        T = n_total / sr
        self.n_blocks = int(np.round(((T - self.T_g) / (self.T_g * self.step)))) + 1
        self.z = np.zeros((channels, self.n_blocks))
        self.j = 0
        self.buf = np.zeros((0, channels))       # K 特性を掛けた信号((N, ch) の C 配列: pyloudnorm と同じ並び)
        self.buf_ofs = 0

    def _bounds(self, j):
        l = int(self.T_g * (j * self.step) * self.sr)
        u = int(self.T_g * (j * self.step + 1) * self.sr)
        return l, u

    def push(self, stereo):
        from scipy.signal import lfilter
        y = np.array(stereo.T, dtype=np.float64, order="C")          # (n, ch)
        for k, (b, a, g) in enumerate(self.stages):
            for c in range(self.ch):
                out, self.zi[k][c] = lfilter(b, a, y[:, c], zi=self.zi[k][c])
                y[:, c] = g * out
        self.buf = np.concatenate([self.buf, y])
        while self.j < self.n_blocks:
            l, u = self._bounds(self.j)
            if u > self.buf_ofs + len(self.buf):
                break
            blk = self.buf[l - self.buf_ofs:u - self.buf_ofs]
            for c in range(self.ch):
                self.z[c, self.j] = (1.0 / (self.T_g * self.sr)) * np.sum(np.square(blk[:, c]))
            self.j += 1
        if self.j < self.n_blocks:                                     # もう使わない先頭を捨てる
            drop = self._bounds(self.j)[0] - self.buf_ofs
            if drop > 0:
                self.buf, self.buf_ofs = self.buf[drop:], self.buf_ofs + drop

    def result(self):
        z, G, j_range = self.z, [1.0, 1.0, 1.0, 1.41, 1.41], np.arange(self.n_blocks)
        l = [-0.691 + 10.0 * np.log10(np.sum([G[i] * z[i, j] for i in range(self.ch)])) for j in j_range]
        J_g = [j for j, l_j in enumerate(l) if l_j >= -70.0]
        z_avg = [np.mean([z[i, j] for j in J_g]) for i in range(self.ch)]
        Gamma_r = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg[i] for i in range(self.ch)])) - 10.0
        J_g = [j for j, l_j in enumerate(l) if (l_j > Gamma_r and l_j > -70.0)]
        z_avg = np.nan_to_num(np.array([np.mean([z[i, j] for j in J_g]) for i in range(self.ch)]))
        return -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg[i] for i in range(self.ch)]))


def process_5p1_stream(input_path, output_path, cfg: PipelineConfig = None, chunk_s=60.0, pad_s=4.0,
                       tmp_path=None, verbose=True):
    """
    長尺(映画 1 本)用。入力を chunk_s 秒ずつ処理し、メモリを一定に保つ。process_5p1 と同じ処理を 2 パスで行う。
      - ゼロ位相フィルタは前後に pad_s 秒の余白を付けて計算し、余白を捨てる(IIR の応答は余白内で倍精度の下限未満まで減衰)
      - BRIR の畳み込みはオーバーラップ加算(残響の尾を次の区間へ繰り越す)
      - 高域シェルフと K 特性フィルタは状態を引き継ぐ
      - 1 パス目は倍精度で一時ファイルへ。2 パス目で音量を決めて書き出す
    """
    from scipy.signal import oaconvolve, sosfilt
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    cfg = cfg or PipelineConfig()
    sr = cfg.sample_rate
    log = print if verbose else (lambda *a: None)
    if cfg.headphone_eq or cfg.virtual_bass:
        raise NotImplementedError("stream 処理は headphone_eq / virtual_bass 未対応")
    info = sf.info(input_path)
    if info.samplerate != sr or info.channels != 6:
        raise ValueError(f"expected 6ch {sr}Hz, got {info.channels}ch {info.samplerate}Hz")
    N = info.frames
    tmp_path = tmp_path or (os.path.splitext(output_path)[0] + ".stream_tmp.w64")
    log(f"cinebinaural (stream): {input_path} -> {output_path}  ({N / sr:.1f}s, 区切り {chunk_s:g}s, 余白 {pad_s:g}s)")

    hrtf = load_hrtf(cfg, sr)
    brirs = build_brirs(hrtf, sr, cfg, log=log)
    names = list(C.SCREEN_CHANNELS + C.SURROUND_CHANNELS)
    if cfg.bass_management:
        brirs = {**brirs, "SUB": build_sub_brir(hrtf, sr, cfg, log=log)}
    M = max(b.shape[1] for b in brirs.values())
    delay = direct_arrival_samples(brirs) if (not cfg.bass_management and cfg.lfe_align) else 0
    out_len = N + M - 1
    tonal = C.TONAL_PRESETS.get(cfg.tonal)
    sos = dsp.rbj_sos("highshelf", tonal[0], tonal[1], tonal[2], sr) if tonal else None
    zi = np.zeros((sos.shape[0], 2, 2)) if tonal else None

    CH, P = int(chunk_s * sr), int(pad_s * sr)
    acc = np.zeros((2, CH + M + delay + 1))
    loud = _StreamingLoudness(sr, out_len)
    peak = 0.0
    written = 0

    def emit(block, fh):
        nonlocal zi, peak, written
        if tonal:
            block, zi = sosfilt(sos, block, axis=-1, zi=zi)
        fh.write(block.T)
        loud.push(block)
        peak = max(peak, float(np.max(np.abs(block))))
        written += block.shape[1]

    with sf.SoundFile(input_path) as fin, sf.SoundFile(tmp_path, "w", sr, 2, "DOUBLE", format="W64") as ftmp:
        for s0 in range(0, N, CH):
            e0 = min(s0 + CH, N)
            pl, pr = min(P, s0), min(P, N - e0)
            fin.seek(s0 - pl)
            x = fin.read(e0 - s0 + pl + pr, dtype="float64", always_2d=True)
            ch = {name: x[:, i] for i, name in enumerate(C.CH_ORDER_5P1)}
            L = e0 - s0
            lfe = dsp.lowpass(ch["LFE"], C.LFE_LP_HZ, sr) * 10 ** (cfg.lfe_gain_db / 20)
            mains = {k: ch[k] for k in names}
            if cfg.tonal == "xcurve":
                for k in mains:
                    hinge = C.XCURVE_SURROUND_HINGE if k in C.SURROUND_CHANNELS else 2000.0
                    mains[k] = dsp.xcurve(mains[k], sr, C.XCURVE_ST202, hinge_hz=hinge)
            if cfg.surround_gain_db:
                for k in C.SURROUND_CHANNELS:
                    mains[k] = mains[k] * 10 ** (cfg.surround_gain_db / 20)
            if cfg.bass_management:
                sub = lfe
                for k in list(mains):
                    low, mains[k] = dsp.crossover_zero_phase(mains[k], cfg.bass_crossover_hz, sr)
                    sub = sub + low
                mains["SUB"] = sub
            for k in mains:                                     # 余白を捨てる
                mains[k] = mains[k][pl:pl + L]
            for name, sig in mains.items():
                for ear in (0, 1):
                    y = oaconvolve(sig, brirs[name][ear])
                    acc[ear, : len(y)] += y
            if not cfg.bass_management:
                acc[:, delay:delay + L] += lfe[pl:pl + L][None, :]
            emit(acc[:, :L].copy(), ftmp)
            acc[:, :-L] = acc[:, L:]
            acc[:, -L:] = 0.0
            log(f"  {e0 / sr:8.1f}s / {N / sr:.1f}s")
        emit(acc[:, : out_len - written].copy(), ftmp)          # 最後の残響の尾

    lufs_in = loud.result()
    gain_db = cfg.target_lufs - lufs_in
    if not cfg.allow_limiting:
        gain_db = min(gain_db, C.LIMIT_CEILING_DBFS - 20 * np.log10(peak + 1e-12))
    g = 10 ** (gain_db / 20)
    ceil = 10 ** (C.LIMIT_CEILING_DBFS / 20)
    w = 2 * max(1, int(sr * C.LIMITER_WINDOW_MS / 1000.0)) + 1
    loud2 = _StreamingLoudness(sr, out_len)
    gr_max = 0.0
    with sf.SoundFile(tmp_path) as ft, sf.SoundFile(output_path, "w", sr, 2, "PCM_24") as fo:
        for s0 in range(0, out_len, CH):
            e0 = min(s0 + CH, out_len)
            pl, pr = (min(w, s0), min(w, out_len - e0)) if cfg.allow_limiting else (0, 0)
            ft.seek(s0 - pl)
            y = ft.read(e0 - s0 + pl + pr, dtype="float64", always_2d=True).T * g
            if cfg.allow_limiting:
                gl = np.minimum(1.0, ceil / np.maximum(np.max(np.abs(y), axis=0), 1e-12))
                if np.any(gl < 1.0):
                    gl = uniform_filter1d(minimum_filter1d(gl, size=w, mode="nearest"), size=w, mode="nearest")
                    y = y * gl[None, :]
                    gr_max = max(gr_max, float(-20 * np.log10(np.min(gl))))
            y = y[:, pl:pl + (e0 - s0)]
            fo.write(y.T)
            loud2.push(y)
    os.remove(tmp_path)
    lim = f"limiter GR max {gr_max:.1f}dB" if cfg.allow_limiting else "limiter 不使用"
    log(f"  out: {loud2.result():.1f} LUFS, {lim}, {out_len / sr:.1f}s")
    return output_path


def render_naive_downmix(input_path, output_path, sr=C.TARGET_SR, target_lufs=C.TARGET_LUFS,
                         verbose=True):
    """A/B 用の基準: 一般的なプレーヤー(ffmpeg 既定)と同じ ITU ダウンミックス(LFE なし)。"""
    ch = _load_5p1(input_path, sr)
    k = 1 / np.sqrt(2)
    stereo = np.stack([ch["FL"] + k * ch["FC"] + k * ch["SL"],
                       ch["FR"] + k * ch["FC"] + k * ch["SR"]])
    stereo, lufs, gr = _finalize(stereo, sr, target_lufs)
    _write(output_path, stereo, sr)
    if verbose:
        print(f"naive downmix: {output_path} ({lufs:.1f} LUFS, limiter GR max {gr:.1f}dB)")
    return output_path


__all__ = ["process_5p1", "render_naive_downmix", "PipelineConfig"]
