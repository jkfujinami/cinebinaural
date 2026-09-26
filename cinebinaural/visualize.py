"""
cinebinaural 可視化: 仮想劇場の箱・スピーカー配置と、各スピーカーに流れている信号の大きさを動画にする。

  - 部屋(ROOM_DIMS)、スクリーン、フロント 3 本、サブ、サラウンド・アレイ(左右の壁)、聴く位置を 3D で描く
  - 各スピーカーの信号は、パイプラインと同じ処理のあとのもの(サラウンドのレベル、LFE +10dB と LPF、
    ベースマネジメントで 5ch の低域 + LFE -> サブ)。アレイは同じ信号を全スピーカーで鳴らす
  - 明るさ・大きさ・聴く位置への線の濃さ = 短時間 RMS(メーター特性: 立ち上がり即時, 戻り 300ms)
  - 音声トラックにはバイノーラル出力を付ける
"""
import os
import subprocess

import numpy as np
import soundfile as sf

from . import config as C
from . import dsp
from .config import PipelineConfig
from .spatial import channel_directions, _channel_speakers

FPS = 30
DB_FLOOR = -60.0          # これ以下は消灯
RELEASE_MS = 300.0
RING_LIFE_S = 1.2          # 波紋 1 つが広がって消えるまで
RING_EVERY_S = 0.27        # 波紋を出す間隔(音が鳴っている間)
RING_ONSET_DB = 4.0        # 1 フレームでこれ以上大きくなったら(打音など)すぐに出す
RING_MAX_M = (1.0, 6.0)    # 最大半径 [m]: 小さい音 -> 大きい音
COLORS = {"FL": "#4fc3f7", "FR": "#4fc3f7", "FC": "#ffd54f", "SL": "#ba68c8", "SR": "#ba68c8", "SUB": "#ef5350"}


def speaker_feeds(input_path, cfg):
    """5.1 -> {スピーカー系統: (モノ信号, [位置...])}。パイプライン(process_5p1)と同じ順の処理。"""
    x, sr = sf.read(input_path, always_2d=True)
    ch = {k: x[:, i] for i, k in enumerate(C.CH_ORDER_5P1)}
    lfe = dsp.lowpass(ch["LFE"], C.LFE_LP_HZ, sr) * 10 ** (cfg.lfe_gain_db / 20)
    feeds, pos = {}, {}
    for name, (az, el) in channel_directions(cfg).items():
        sig = ch[name]
        if name in C.SURROUND_CHANNELS and cfg.surround_gain_db:
            sig = sig * 10 ** (cfg.surround_gain_db / 20)
        feeds[name] = sig
        pos[name] = [p for p, _, _ in _channel_speakers(name, az, el, cfg)]
    if cfg.bass_management:
        sub = lfe
        for k in list(feeds):
            low, feeds[k] = dsp.crossover_zero_phase(feeds[k], cfg.bass_crossover_hz, sr)
            sub = sub + low
    else:
        sub = lfe
    feeds["SUB"] = sub
    pos["SUB"] = [np.array(C.SUB_POS_ON_FLOOR if cfg.lf_boundary else C.SUB_POS, dtype=float)]
    return feeds, pos, sr, x.shape[0]


def meter_levels(sig, sr, n_frames, fps=FPS):
    """フレームごとの短時間 RMS [dBFS](窓 50ms)にメーター特性(戻り RELEASE_MS)を掛ける。"""
    hop = sr / fps
    win = int(0.05 * sr)
    p = np.concatenate([np.zeros(win), sig ** 2])
    cs = np.concatenate([[0.0], np.cumsum(p)])
    out = np.empty(n_frames)
    rel = np.exp(-1.0 / (RELEASE_MS / 1000 * fps)) if RELEASE_MS > 0 else 0.0
    prev = DB_FLOOR
    for k in range(n_frames):
        e = int(min((k + 1) * hop, len(sig))) + win
        rms = np.sqrt(max(cs[e] - cs[max(e - win, 0)], 0.0) / win)
        db = max(20 * np.log10(rms + 1e-12), DB_FLOOR)
        prev = db if db > prev else DB_FLOOR + (prev - DB_FLOOR) * rel
        out[k] = prev
    return out


def ring_events(lv, fps=FPS):
    """メーター値の列 -> [(出たフレーム, 強さ 0..1)]。一定間隔 + 立ち上がり(打音)で出す。"""
    every, events, last = max(int(round(RING_EVERY_S * fps)), 1), [], -10 ** 9
    for k in range(len(lv)):
        t = (lv[k] - DB_FLOOR) / -DB_FLOOR
        onset = k > 0 and lv[k] - lv[k - 1] > RING_ONSET_DB and k - last >= 3
        if t > 0.05 and (k - last >= every or onset):
            events.append((k, t))
            last = k
    return events


def _ring_plane(p, is_sub):
    """波紋を描く面の 2 軸(スピーカーの付いている壁の面。サブは床)。"""
    if is_sub:
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    W, D, _ = C.ROOM_DIMS
    d = {0: min(p[0], W - p[0]), 1: min(p[1], D - p[1])}
    normal = min(d, key=d.get)
    return (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])) if normal == 0 else \
        (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))


def _draw_room(ax):
    W, D, H = C.ROOM_DIMS
    for a, b in (((0, 0, 0), (W, 0, 0)), ((0, D, 0), (W, D, 0)), ((0, 0, H), (W, 0, H)), ((0, D, H), (W, D, H)),
                 ((0, 0, 0), (0, D, 0)), ((W, 0, 0), (W, D, 0)), ((0, 0, H), (0, D, H)), ((W, 0, H), (W, D, H)),
                 ((0, 0, 0), (0, 0, H)), ((W, 0, 0), (W, 0, H)), ((0, D, 0), (0, D, H)), ((W, D, 0), (W, D, H))):
        ax.plot(*zip(a, b), color="#3a3f4b", lw=0.8)
    # スクリーン(前の壁の大部分。IMAX は壁一面に近い)
    sx0, sx1, sz0, sz1 = 1.5, W - 1.5, 1.0, H - 1.0
    ax.plot([sx0, sx1, sx1, sx0, sx0], [0.05] * 5, [sz0, sz0, sz1, sz1, sz0], color="#cfd8dc", lw=1.2, alpha=0.6)
    # 床の格子
    for gx in np.arange(0, W + 0.1, 3):
        ax.plot([gx, gx], [0, D], [0, 0], color="#22262e", lw=0.5)
    for gy in np.arange(0, D + 0.1, 3):
        ax.plot([0, W], [gy, gy], [0, 0], color="#22262e", lw=0.5)


def render_video(input_5p1, binaural_wav, out_mp4, cfg=None, fps=FPS, size=(1280, 720), view=(35, 90),
                 verbose=True, preview_frame=None, style="ripple", lang="ja"):
    """
    style: "ripple" = 各スピーカーから線の円が波紋のように広がる(音が大きいほど遠くまで・濃く)/
           "rays" = スピーカーの大きさと聴く位置への線の濃さで示す(最初の版)。
    preview_frame を渡すと、そのフレームだけを out_mp4(.png)に保存して終わる(見た目の確認用)。
    lang: 画面の文字 "ja" / "en"
    """
    T = {"ja": dict(seat="聴く位置", meter="dBFS(スピーカーへの信号)",
                    room="仮想劇場 {W:g}×{D:g}×{H:g} m / サラウンドは左右の壁のアレイ(各 {n} 本)"),
         "en": dict(seat="listener", meter="dBFS (speaker feed)",
                    room="Virtual theater {W:g}×{D:g}×{H:g} m / surround arrays on side walls ({n} per side)")}[lang]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]

    cfg = cfg or PipelineConfig()
    log = print if verbose else (lambda *a: None)
    feeds, pos, sr, n = speaker_feeds(input_5p1, cfg)
    n_frames = int(np.ceil(n / sr * fps))
    levels = {k: meter_levels(v, sr, n_frames, fps) for k, v in feeds.items()}
    log(f"visualize: {input_5p1} ({n / sr:.1f}s, {n_frames} frames) -> {out_mp4}")

    Lp = np.array(C.LISTENER_POS)
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor="#0d0f14")
    ax = fig.add_axes([0.0, 0.0, 0.78, 1.0], projection="3d", facecolor="#0d0f14")
    axm = fig.add_axes([0.80, 0.12, 0.18, 0.76], facecolor="#0d0f14")
    W, D, H = C.ROOM_DIMS
    ax.set_xlim(0, W)
    ax.set_ylim(0, D)
    ax.set_zlim(0, H)
    ax.set_box_aspect((W, D, H), zoom=1.05)
    ax.set_proj_type("persp", focal_length=0.6)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    _draw_room(ax)
    ax.scatter(*Lp, s=90, c="#ffffff", marker="o", depthshade=False)
    ax.text(Lp[0], Lp[1] + 1.2, Lp[2] - 1.5, T["seat"], color="#ffffff", fontsize=9, ha="center")
    for name, ps in pos.items():
        p = ps[0] if len(ps) == 1 else ps[len(ps) // 2]
        off = (0, 1.3, 1.6) if name != "SUB" else (0, 1.3, -1.3)
        ax.text(p[0] + off[0], p[1] + off[1], p[2] + off[2], name, color=COLORS[name], fontsize=9, ha="center")

    order = ["FL", "FC", "FR", "SL", "SR", "SUB"]
    dyn = []
    rays = []
    rings = []                                   # (系統, スピーカー位置, 面の 2 軸, 線オブジェクトの束)
    events = {k: ring_events(levels[k], fps) for k in order}
    life = int(round(RING_LIFE_S * fps))
    pool = life // 3 + 2
    theta = np.linspace(0, 2 * np.pi, 97)
    W, D, H = C.ROOM_DIMS
    for name in order:
        ps = np.array(pos[name])
        sc = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=20, c=COLORS[name], marker="s", depthshade=False)
        dyn.append((name, sc, ps))
        for p in ps:
            if style == "rays":
                ln, = ax.plot([p[0], Lp[0]], [p[1], Lp[1]], [p[2], Lp[2]], color=COLORS[name], lw=1.2, alpha=0.0)
                rays.append((name, ln))
            else:
                u, v = _ring_plane(p, name == "SUB")
                c = p + (np.array([0, 0, 0.05 - p[2]]) if name == "SUB" else 0)
                lines = [ax.plot([], [], [], color=COLORS[name], lw=1.6, alpha=0.0)[0] for _ in range(pool)]
                rings.append((name, c, u, v, lines))

    def ring_state(name, k):
        """フレーム k で見えている波紋 [(半径, 濃さ)]。"""
        out = []
        for k0, t0 in reversed(events[name]):
            age = (k - k0) / life
            if age < 0:
                continue
            if age >= 1:
                break
            rmax = RING_MAX_M[0] + (RING_MAX_M[1] - RING_MAX_M[0]) * t0 ** 1.5
            out.append((rmax * (1 - (1 - age) ** 2), min(1.0, 1.1 * t0 ** 0.6 * (1 - age) ** 1.3)))
        return out
    bars = axm.barh(range(len(order)), [0] * len(order), color=[COLORS[k] for k in order], height=0.6)
    axm.set_yticks(range(len(order)), order, color="#cfd8dc", fontsize=9)
    axm.invert_yaxis()
    axm.set_xlim(DB_FLOOR, 0)
    axm.set_xticks([-60, -40, -20, 0], ["-60", "-40", "-20", "0"], color="#90a4ae", fontsize=8)
    axm.set_xlabel(T["meter"], color="#90a4ae", fontsize=8)
    for s in axm.spines.values():
        s.set_color("#3a3f4b")
    axm.tick_params(colors="#90a4ae")
    title = fig.text(0.02, 0.95, "", color="#eceff1", fontsize=12)
    fig.text(0.02, 0.91, T["room"].format(W=W, D=D, H=H, n=C.SURROUND_ARRAY_N),
             color="#90a4ae", fontsize=9)

    def draw(k):
        for (name, sc, ps) in dyn:
            t = (levels[name][k] - DB_FLOOR) / -DB_FLOOR
            sc.set_sizes(np.full(len(ps), (15 + 260 * t ** 2) if style == "rays" else (18 + 50 * t)))
            sc.set_alpha(0.25 + 0.75 * t)
        for name, ln in rays:
            t = (levels[name][k] - DB_FLOOR) / -DB_FLOOR
            ln.set_alpha(0.7 * t ** 3)
        for name, c, u, v, lines in rings:
            st = ring_state(name, k)
            for j, ln in enumerate(lines):
                if j >= len(st):
                    ln.set_alpha(0.0)
                    continue
                r, a = st[j]
                pts = c[:, None] + r * (np.cos(theta)[None, :] * u[:, None] + np.sin(theta)[None, :] * v[:, None])
                out = ((pts[0] < 0) | (pts[0] > W) | (pts[1] < 0) | (pts[1] > D) | (pts[2] < 0) | (pts[2] > H))
                pts[:, out] = np.nan
                ln.set_data_3d(pts[0], pts[1], pts[2])
                ln.set_alpha(a)
        for b, name in zip(bars, order):
            b.set_width(levels[name][k] - DB_FLOOR)
            b.set_x(DB_FLOOR)
        sec = k / fps
        title.set_text(f"{int(sec // 60):d}:{sec % 60:05.2f}")
        fig.canvas.draw()

    if preview_frame is not None:
        draw(preview_frame)
        fig.savefig(out_mp4, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_mp4

    tmp = out_mp4 + ".video.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{size[0]}x{size[1]}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n_frames):
        draw(k)
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if verbose and k % (fps * 10) == 0:
            log(f"  frame {k}/{n_frames}")
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", binaural_wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", out_mp4], check=True)
    os.remove(tmp)
    log(f"  -> {out_mp4}")
    return out_mp4


__all__ = ["render_video", "speaker_feeds", "meter_levels"]
