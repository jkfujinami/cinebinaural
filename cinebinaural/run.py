"""
cinebinaural CLI。

  # m2ts から TrueHD 5.1 を抽出 (時刻は秒 / mm:ss / hh:mm:ss)
  python -m cinebinaural.run extract movie.m2ts cinebinaural/data/clip_5p1.wav --start 29:11 --end 31:31

  # 5.1 wav -> IMAX バイノーラル
  python -m cinebinaural.run process cinebinaural/data/clip_5p1.wav cinebinaural/out/clip_binaural.wav

  # 比較用のバリアント一式(LUFS 揃え)
  python -m cinebinaural.run variants cinebinaural/data/clip_5p1.wav cinebinaural/out
"""
import argparse
import dataclasses
import os
import subprocess

from .config import PipelineConfig, PRESETS
from .pipeline import process_5p1, process_5p1_lowmem, render_naive_downmix

# 聴き比べ用のバリアント(v2 を基準に 1 項目ずつ変える)
VARIANTS = {
    "v1": dict(room_model="v1"),                         # 前回の聴感ベスト
    "v2": dict(),                                        # 部屋モデル v2(既定)
    "v2_array": dict(surround_array=True),               # サラウンドをアレイ化
    "v2_thx": dict(tonal="thx"),                         # THX Re-EQ 相当のシェルフ
    "v2_kemar": dict(hrtf="kemar"),                      # KEMAR ダミーヘッド
    "v2_wide": dict(front_angle=37.0),                   # フロント L/R をスクリーン端相当へ
    # 2026-09-25 の比較実験の条件(エコー / メリハリの原因切り分け)
    "v2_tailfix": dict(late_onset_ms=(12.0, 22.0)),      # 残響を 12ms から単調減衰で開始
    "v2_rear099": dict(rear_wall_alpha=0.99),            # 後壁の吸音率 0.99
    "v2_tailfix_rear099": dict(late_onset_ms=(12.0, 22.0), rear_wall_alpha=0.99),
}


# 聴感で選ばれたベース(tailfix / tailfix_rear099)に、各オプションを掛け合わせる
_MODIFIERS = {
    "wide": dict(front_angle=37.0),
    "thx": dict(tonal="thx"),
    "array": dict(surround_array=True),
    "kemar": dict(hrtf="kemar"),
}
for _base in ("v2_tailfix", "v2_tailfix_rear099"):
    for _name, _over in _MODIFIERS.items():
        VARIANTS[f"{_base}_{_name}"] = {**VARIANTS[_base], **_over}
VARIANTS["v2_tailfix_rear099_wide_array"] = {**VARIANTS["v2_tailfix_rear099"],
                                             **_MODIFIERS["wide"], **_MODIFIERS["array"]}

# ここまでは LFE の時間合わせ(2026-09-25 修正)より前に書き出した。既存ファイルを再現できるよう修正前で固定する。
for _v in VARIANTS.values():
    _v.setdefault("lfe_align", False)
VARIANTS["v2_tailfix_rear099_wide_array_lfealign"] = {**VARIANTS["v2_tailfix_rear099_wide_array"],
                                                      "lfe_align": True}
# ベースマネジメントのテスト(今のベストとの比較用)
VARIANTS["v2_tailfix_rear099_wide_array_bm"] = {**VARIANTS["v2_tailfix_rear099_wide_array_lfealign"],
                                                "bass_management": True}
VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3"] = {**VARIANTS["v2_tailfix_rear099_wide_array_bm"],
                                                      "surround_gain_db": -3.0}
VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15"] = {**VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3"],
                                                             "surround_elevation_deg": 15.0,
                                                             "hrtf_el_interp": True}
# 低音の見直し(Toole 2015 / Dolby Atmos Spec): ① 境界効果、② 低域の残響を伸ばさない
VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd"] = {
    **VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15"], "lf_boundary": True}
VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf"] = {
    **VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd"], "lf_rt_factor": 1.0}
# ③ 校正の基準を直接音に(Toole 2015)。①②と合わせて適用
VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf_direct"] = {
    **VARIANTS["v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf"], "timbre_reference": "direct"}
# ①②③をベースに、問題を 1 つずつ直した版
_B = "v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf_direct"
VARIANTS[_B + "_capref"] = {**VARIANTS[_B], "cap_early_energy": True}           # サブの低音の重さ
VARIANTS[_B + "_xcurve"] = {**VARIANTS[_B], "tonal": "xcurve"}                  # 高域: 映画館の X カーブ
VARIANTS[_B + "_thx"] = {**VARIANTS[_B], "tonal": "thx"}                        # 高域: THX Re-EQ
VARIANTS[_B + "_lf20"] = {**VARIANTS[_B], "neutral_low_hz": 20.0, "tail_lowcut_hz": 15.0}  # 20-40Hz の補正 → ③のもとでは効果なし
VARIANTS[_B + "_subeq"] = {**VARIANTS[_B], "sub_room_eq": True}                # サブの低域の山を削る(ノッチ)
VARIANTS[_B + "_capref_xcurve_subeq"] = {**VARIANTS[_B], "cap_early_energy": True, "tonal": "xcurve",
                                          "sub_room_eq": True}
VARIANTS[_B + "_capref_subeq"] = {**VARIANTS[_B], "cap_early_energy": True, "sub_room_eq": True}   # 高域は mild のまま
# ③(直接音基準)なし: ①②のベースに capref + subeq
_B12 = "v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf"
VARIANTS[_B12 + "_capref_subeq"] = {**VARIANTS[_B12], "cap_early_energy": True, "sub_room_eq": True}
# サラウンドの -3dB を外した版
for _t in ("v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd", "v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf"):
    VARIANTS[_t + "_surr0"] = {**VARIANTS[_t], "surround_gain_db": 0.0}

# 2026-09-25 に既定値を v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf に切り替えた。
# ここまでのバリアントは古い既定値からの差分として定義しているので、古い既定値を固定して
# 既存ファイルを同じように作り直せるようにする。
LEGACY_DEFAULTS = dict(
    surround_array=False, surround_elevation_deg=0.0, hrtf_el_interp=False, late_onset_ms=None,
    rear_wall_alpha=None, lf_boundary=False, lf_rt_factor=1.5, front_angle=30.0,
    surround_gain_db=0.0, bass_management=False,
)
for _k in list(VARIANTS):
    VARIANTS[_k] = {**LEGACY_DEFAULTS, **VARIANTS[_k]}
VARIANTS["default"] = {}   # 現在の既定の構成(標準)
VARIANTS["standard2"] = dict(PRESETS["standard2"])
VARIANTS["standard3"] = dict(PRESETS["standard3"])
VARIANTS["standard4"] = dict(PRESETS["standard4"])

def _seconds(s):
    v = 0.0
    for part in str(s).split(":"):
        v = v * 60 + float(part)
    return v


def _extract(m2ts, out_wav, start, dur, stream):
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", m2ts, "-map", f"0:{stream}",
           "-t", str(dur), "-c:a", "pcm_s24le", out_wav, "-loglevel", "error"]
    subprocess.run(cmd, check=True)
    print(f"extracted: {out_wav} ({start:.0f}s + {dur:.0f}s)")


def _cfg(a):
    # プリセットと重なる項目(hrtf, tonal, hf_peak_cut など)は CLI で明示したときだけ上書きする
    given = {k: v for k, v in dict(
        hrtf=a.hrtf, tonal=a.tonal, hf_peak_cut=a.hf_peak_cut,
        hrtf_symmetric=None if a.hrtf_symmetric is None else {"off": False, "env": True}.get(a.hrtf_symmetric, a.hrtf_symmetric),
        hrtf_az_interp=a.hrtf_az_interp,
    ).items() if v is not None}
    cfg = PipelineConfig(
        **{**PRESETS[a.preset], **given},
        ear_balance=not a.no_ear_balance,
        room_model=a.room,
        rt60_mid=a.rt60,
        image_order=a.image_order,
        surround_array=a.surround_array,
        front_angle=a.front_angle,
        timbre_neutral_strength=a.timbre,
        lfe_gain_db=a.lfe_gain_db,
        virtual_bass=a.virtual_bass,
        headphone_eq=a.headphone_eq,
        target_lufs=a.target_lufs,
        allow_limiting=not a.no_limit,
    )
    if a.simple:   # 標準 5.1 位置の HRIR を畳み込むだけ
        cfg = dataclasses.replace(cfg, room_model="none", timbre_neutral_strength=0.0, tonal="off")
    return cfg


def _render_args(p):
    d = PipelineConfig()
    p.add_argument("--preset", default="standard", choices=list(PRESETS),
                   help="standard=標準 / standard2=標準+サブの低音を締める / standard3=標準2+直接音基準の校正 / standard4=標準3+KU100+高域シェルフなし+高域の細い山を削る")
    p.add_argument("--hrtf", default=None, help="riec / kemar / ku100 / irc1040 / SOFA のパス(既定はプリセットの値)")
    p.add_argument("--no-ear-balance", action="store_true", help="HRTF の左右耳補正を外す")
    p.add_argument("--room", default=d.room_model, choices=["v2", "v1", "none"])
    p.add_argument("--rt60", type=float, default=d.rt60_mid, help="500Hz の RT60 [s]")
    p.add_argument("--image-order", type=int, default=d.image_order)
    p.add_argument("--surround-array", action=argparse.BooleanOptionalAction, default=d.surround_array,
                   help="サラウンドをアレイで鳴らす(--no-surround-array で点音源)")
    p.add_argument("--front-angle", type=float, default=d.front_angle)
    p.add_argument("--timbre", type=float, default=d.timbre_neutral_strength, help="音色中立化 0..1")
    p.add_argument("--lfe-gain-db", type=float, default=d.lfe_gain_db)
    p.add_argument("--virtual-bass", action="store_true", help="低域の弱いヘッドホン向け")
    p.add_argument("--tonal", default=None, choices=["mild", "thx", "xcurve", "off"], help="既定はプリセットの値")
    p.add_argument("--headphone-eq", default=None, help="AutoEq ParametricEQ.txt")
    p.add_argument("--target-lufs", type=float, default=d.target_lufs)
    p.add_argument("--no-limit", action="store_true",
                   help="リミッターを使わない(ピークが -1dBFS に収まるところまで音量を下げる)")
    p.add_argument("--simple", action="store_true", help="直接音の HRIR 畳み込みのみ")
    p.add_argument("--hf-peak-cut", action="store_true", default=None, help="直接音の 5-16kHz の細い山を削る(谷は残す)")
    p.add_argument("--hrtf-symmetric", default=None, choices=["off", "env", "left", "right"],
                   help="HRTF の左右対称化: env=1/3oct の輪郭だけ / left・right=その耳を鏡映しにして両耳に使う")
    p.add_argument("--hrtf-az-interp", action="store_true", default=None, help="方位角方向にも補間(15° 刻みなど粗い HRTF 用)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cinebinaural")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="m2ts から 5.1 を抽出")
    pe.add_argument("m2ts")
    pe.add_argument("out_wav")
    pe.add_argument("--start", default="0")
    pe.add_argument("--end", default=None)
    pe.add_argument("--dur", default=None)
    pe.add_argument("--stream", type=int, default=2, help="TrueHD 5.1 のストリーム番号")

    pp = sub.add_parser("process", help="5.1 wav -> IMAX バイノーラル")
    pp.add_argument("input_wav")
    pp.add_argument("output_wav")
    pp.add_argument("--lowmem", action="store_true",
                    help="長尺用: 区切らずに同じ計算をメモリを抑えて行う(出力は通常と同一)")
    pp.add_argument("--work-dir", default=None, help="--lowmem の一時退避先")
    _render_args(pp)

    pa = sub.add_parser("ab", help="指定設定と素のダウンミックスを LUFS 揃えで書き出す")
    pa.add_argument("input_wav")
    pa.add_argument("out_dir")
    pa.add_argument("--tag", default="cinebinaural")
    _render_args(pa)

    pu = sub.add_parser("upmix", help="ステレオ -> 5.1(AI 音源分離 + センター/響きの抽出)")
    pu.add_argument("stereo_wav")
    pu.add_argument("out_5p1_wav")
    pu.add_argument("--center", default="vocals", choices=["vocals", "repan"],
                    help="vocals=ボーカルの中央だけ FC へ / repan=全ステムを L・C・R に振り直す(Avendano & Jot 2004)")
    pu.add_argument("--ambience", default="mask", choices=["mask", "ls"], help="mask=1-コヒーレンス / ls=最小二乗(Faller 2006)")
    pu.add_argument("--surround-decorr", action="store_true", help="サラウンドを無相関化(ランダム位相, 200Hz 以上)")
    pu.add_argument("--surround-delay-ms", type=float, default=0.0, help="サラウンドの遅延(先行音効果, 例 10)")
    pu.add_argument("--stems-cache", default=None, help="Demucs の分離結果を保存・再利用する .npz")

    pz = sub.add_parser("video", help="仮想劇場のスピーカー配置と各スピーカーの信号の大きさを動画にする")
    pz.add_argument("input_wav", help="5.1 wav(スピーカーの信号を作る元)")
    pz.add_argument("binaural_wav", help="音声トラックに付けるバイノーラル出力")
    pz.add_argument("out_mp4")
    pz.add_argument("--view", type=float, nargs=2, default=(35.0, 90.0), metavar=("ELEV", "AZIM"),
                    help="カメラの仰角・方位角(90 = 後ろから)")
    pz.add_argument("--lang", default="ja", choices=["ja", "en"], help="画面の文字の言語")
    pz.add_argument("--style", default="ripple", choices=["ripple", "rays"], help="ripple=波紋 / rays=大きさと線")
    _render_args(pz)
    pv = sub.add_parser("variants", help="比較用バリアント一式を LUFS 揃えで書き出す")
    pv.add_argument("input_wav")
    pv.add_argument("out_dir")
    pv.add_argument("--only", nargs="*", default=None, help=f"一部だけ: {' '.join(VARIANTS)}")
    pv.add_argument("--target-lufs", type=float, default=PipelineConfig().target_lufs)

    a = p.parse_args(argv)
    stem = os.path.splitext(os.path.basename(getattr(a, "input_wav", "") or ""))[0]
    if a.cmd == "extract":
        start = _seconds(a.start)
        dur = _seconds(a.end) - start if a.end else _seconds(a.dur or 90)
        _extract(a.m2ts, a.out_wav, start, dur, a.stream)
    elif a.cmd == "process":
        os.makedirs(os.path.dirname(a.output_wav) or ".", exist_ok=True)
        if a.lowmem:
            process_5p1_lowmem(a.input_wav, a.output_wav, _cfg(a), work_dir=a.work_dir)
        else:
            process_5p1(a.input_wav, a.output_wav, _cfg(a))
    elif a.cmd == "ab":
        os.makedirs(a.out_dir, exist_ok=True)
        process_5p1(a.input_wav, os.path.join(a.out_dir, f"{stem}_{a.tag}.wav"), _cfg(a))
        render_naive_downmix(a.input_wav, os.path.join(a.out_dir, f"{stem}_naive.wav"),
                             target_lufs=a.target_lufs)
    elif a.cmd == "upmix":
        from .upmix import upmix_stereo_to_5p1
        os.makedirs(os.path.dirname(a.out_5p1_wav) or ".", exist_ok=True)
        upmix_stereo_to_5p1(a.stereo_wav, a.out_5p1_wav, center=a.center, ambience=a.ambience,
                            surround_decorr=a.surround_decorr, surround_delay_ms=a.surround_delay_ms,
                            stems_cache=a.stems_cache)
    elif a.cmd == "video":
        from .visualize import render_video
        render_video(a.input_wav, a.binaural_wav, a.out_mp4, _cfg(a), view=tuple(a.view), style=a.style, lang=a.lang)
    elif a.cmd == "variants":
        os.makedirs(a.out_dir, exist_ok=True)
        for tag, over in VARIANTS.items():
            if a.only and tag not in a.only:
                continue
            cfg = PipelineConfig(target_lufs=a.target_lufs, **over)
            process_5p1(a.input_wav, os.path.join(a.out_dir, f"{stem}_{tag}.wav"), cfg)
        naive = os.path.join(a.out_dir, f"{stem}_naive.wav")
        if not os.path.exists(naive):
            render_naive_downmix(a.input_wav, naive, target_lufs=a.target_lufs)


if __name__ == "__main__":
    main()
