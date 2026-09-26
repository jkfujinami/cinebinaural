"""
cinebinaural 設定・定数。

■ 設計原則(マルチチャンネル・マスター入力) — THEORY_FOUNDATION.md Part 7
  レンダラーの仕事は「部屋」と「頭」を付けることであり、ミックスを作り直すことではない。
  1. 音色はマスターのまま: 各チャンネルの BRIR を音色中立化(両耳平均パワー応答を平坦化)。
  2. ダイナミクス処理・サチュレーション・倍音合成は既定 OFF。
  3. LFE は +10dB(SMPTE RP 200 / Dolby の in-band gain)。
  4. 部屋は物理モデルで作る(部屋モデル v2):
       - 3 次までの image source(各反射を到来方向の HRIR で描く)
       - 面の吸音は素材表の周波数特性を保ったまま、目標 RT60(f) に一致するよう Eyring 式で較正
       - スピーカー指向性(シネマ用ホーン)と空気吸収(ISO 9613-1)
       - 拡散残響は平均自由行程から立ち上がり、知覚的ミキシングタイムで完全に拡散
       - 直接音 / 残響の比は臨界距離から周波数ごとに算出(手置きの DRR 定数は持たない)
  5. A/B 比較は必ずラウドネス(LUFS)を揃える。

5.1(side) の並び (TrueHD / ffmpeg): FL, FR, FC, LFE, SL, SR
HRTF 方位角(SOFA SimpleFreeFieldHRIR): 0=前, +90=左, 270=右 / 仰角 0=水平, +90=真上
"""
from dataclasses import dataclass
from typing import Optional

TARGET_SR = 48000

CH_ORDER_5P1 = ["FL", "FR", "FC", "LFE", "SL", "SR"]

HRTF_PRESETS = {
    "riec": "RIEC_hrir_subject_005.sofa",               # 人間の被験者(左右差を補正して使う)
    "kemar": "D2_HRIR_SOFA/D2_48K_24bit_256tap_FIR_SOFA.sofa",  # SADIE II D2 = KEMAR ダミーヘッド
    "ku100": "KU100_HRIR_FULL2DEG.sofa",                # Neumann KU 100 ダミーヘッド(TH Köln, Bernschütz 2013, 2° 刻み, CC BY 3.0)
    "irc1040": "IRC_1040_R_44100.sofa",                 # IRCAM LISTEN 1040(人間, 44.1kHz・15° 刻み。48kHz 変換と方位補間を使う)
}

# ---- 配置 ----
FRONT_ANGLE = 30.0           # FL/FR の方位角。ITU-R BS.775 は ±30°。IMAX 幅のスクリーン端は ±35〜39°
SURROUND_ANGLE = 110.0       # ITU-R BS.775: 100〜120°
SCREEN_CHANNELS = ("FL", "FR", "FC")
SURROUND_CHANNELS = ("SL", "SR")

# ---- シアター幾何 ----
# x=幅, y=奥行(スクリーンは y=0), z=高さ [m]。寸法は旧 imax/core/constants.py の値(出典なし)。
ROOM_DIMS = (30.0, 24.0, 18.0)
LISTENER_POS = (15.0, 16.0, 6.0)   # 後方 2/3、スタジアム席で目線 ≒ スクリーン中心
SPEAKER_WALL_MARGIN = 0.5
SPEED_OF_SOUND = 343.0
IMAGE_ORDER = 3

# ---- 残響時間 ----
RT60_MID = 0.5               # WSDG: IMAX 仕様 RT60 = 0.5 s(IMAX ブエノスアイレス、450 席)
# Dolby Technical Guidelines (1994) 図 4.4 推奨線: 500Hz の RT に掛ける倍率(周波数は Hz)
RT_FREQ_FACTOR = ((31.5, 1.5), (250.0, 1.0), (2000.0, 1.0), (16000.0, 0.5))

# ---- 面の素材(相対的な周波数特性。絶対値は RT60 目標に較正する) ----
# 吸音率 α(125/250/500/1k/2k/4k Hz)。imax/core/constants.py の素材表。
MATERIALS = {
    "carpet": (0.03, 0.04, 0.09, 0.30, 0.40, 0.51),
    "theater_seats": (0.25, 0.35, 0.45, 0.55, 0.60, 0.60),
    "concrete": (0.01, 0.01, 0.02, 0.02, 0.02, 0.03),
    "acoustic_panel": (0.22, 0.57, 0.92, 0.94, 0.97, 1.00),
    "perforated_screen": (0.15, 0.25, 0.40, 0.50, 0.55, 0.55),
}
MATERIAL_FREQS = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
# (軸, 0=座標0側 / 1=座標max側) -> (名前, {素材: 面積比})。左 = +x(スクリーンを向いたとき)。
SURFACES = {
    (0, 0): ("right_wall", {"concrete": 0.3, "acoustic_panel": 0.7}),
    (0, 1): ("left_wall", {"concrete": 0.3, "acoustic_panel": 0.7}),
    # スクリーン + 背後の吸音バリア [US20050248726A1 §0051]
    (1, 0): ("front_wall", {"perforated_screen": 0.5, "acoustic_panel": 0.5}),
    (1, 1): ("rear_wall", {"acoustic_panel": 1.0}),
    (2, 0): ("floor", {"theater_seats": 1.0}),
    (2, 1): ("ceiling", {"acoustic_panel": 1.0}),
}

# ---- スピーカー指向性 ----
# シネマ用スクリーンスピーカー: -6dB で水平 90〜100° x 垂直 40〜50°、DI 10dB [US6513622]
# (h, v) は -6dB になる片側角度 [deg]。低域は無指向、DIRECTIVITY_FREQS の間で指向性が立ち上がる。
SCREEN_BEAM = (45.0, 22.5)
SURROUND_BEAM = (60.0, 60.0)
DIRECTIVITY_FREQS = (200.0, 1000.0)
DIRECTIVITY_FLOOR_DB = -25.0

# ---- サラウンド・アレイ(任意) ----
# Dolby TG §2.3.1: サラウンドは点音源に聞こえてはならない。側壁の中ほどから後壁まで均等に並べる。
SURROUND_ARRAY_N = 6

# ---- 後期残響 ----
LATE_COHERENT_BELOW_HZ = 300.0  # これ以下は両耳でコヒーレント(拡散音場の両耳相関)
BAND_CENTERS = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)

# ---- 音色中立化 ----
TIMBRE_NEUTRAL_STRENGTH = 1.0
TIMBRE_SMOOTH_OCT = 1 / 3
HF_PEAK_CUT_HZ = (5000.0, 16000.0)   # P1(約 4kHz)より上だけを対象にする
HF_PEAK_FINE_OCT = 1 / 12

# ---- ベースマネジメント(任意) ----
# IMAX のサブベースは「5 チャンネルから取り出した低域」を再生し、スクリーン中央の下に置く [US20050248726A1 §0054-0055]
SUB_POS = (15.0, 0.5, 1.0)
SUB_POS_ON_FLOOR = (15.0, 0.5, 0.3)   # lf_boundary 時: 床に置く(キャビネット中心の高さ)
# 低域の境界効果による指向係数 [Dolby Atmos Spec §2 のバッフル壁 / 半空間, Toole 2015: バッフル壁は 2π]
LF_BOUNDARY_Q = {"screen": 2.0, "surround": 2.0, "sub": 4.0}
BASS_CROSSOVER_HZ = 80.0     # 標準的なベースマネジメントのクロスオーバー(THX も 80Hz)

# ---- LFE ----
LFE_GAIN_DB = 10.0
LFE_LP_HZ = 120.0

# ---- 任意: 低域の弱いヘッドホン向け missing fundamental(既定 OFF) ----
VB_CROSSOVER_HZ = 60.0
VB_HARMONIC_LO = 60.0
VB_HARMONIC_HI = 150.0
VB_GAIN = 0.3

# ---- 知覚的トーナル(X-Curve を literally 適用しない) ----
# "mild": 4kHz 中心 -2dB(先行調査レポート由来)
# "thx" : THX Re-EQ 相当。4〜5kHz まで平坦、10kHz で約 -4dB、以降シェルフ
# SMPTE ST 202:2010 表1(200-500 席の中規模劇場)の X カーブ高域部分 [Hz, dB]
XCURVE_ST202 = ((2000, 0), (2500, -1), (3150, -2), (4000, -3), (5000, -4), (6300, -5),
                (8000, -6), (10000, -7), (12500, -9), (16000, -11), (20000, -13))
XCURVE_SURROUND_HINGE = 4000.0   # ST 202 Annex A.5f: サラウンドは曲がり始めを 4kHz まで上げてよい
TONAL_PRESETS = {
    "mild": (4000.0, -2.0, 0.7071),
    "thx": (8000.0, -4.5, 0.9),
}

# ---- 出力 ----
TARGET_LUFS = -16.0
LIMIT_CEILING_DBFS = -1.0
LIMITER_WINDOW_MS = 15.0


# 名前付きプリセット(PipelineConfig の既定値からの差分)
PRESETS = {
    # 標準: v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf(2026-09-25 確定)
    "standard": {},
    # 標準2: 標準 + サブの初期反射の上限(cap_early_energy)+ ルームノードのノッチ(sub_room_eq)
    #        = v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf_capref_subeq。低音の尾が短く締まる
    "standard2": {"cap_early_energy": True, "sub_room_eq": True},
    # 標準3: 標準2 + 音色中立化を直接音基準に(Toole 2015)
    #        = v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf_direct_capref_subeq。低音の直接音が前に出る
    "standard3": {"cap_early_energy": True, "sub_room_eq": True, "timbre_reference": "direct"},
    # 標準4: 標準3 + KU 100 ダミーヘッド(左右対称)+ mild シェルフなし(直接音基準と高域の扱いを Toole 2015 で統一)
    #        + 直接音の 5-16kHz の細い山を削る(谷は残す, Iida 2007)。2026-09-27、2:13:03-2:18:03 等の試聴で決定
    "standard4": {"cap_early_energy": True, "sub_room_eq": True, "timbre_reference": "direct",
                  "hrtf": "ku100", "tonal": "off", "hf_peak_cut": True},
}


@dataclass
class PipelineConfig:
    """
    既定値 = 聴感評価で選んだ構成(2026-09-25): 部屋モデル v2 / 残響 12ms から単調減衰 / 後壁 α0.99 /
    フロント ±37° / サラウンド・アレイ(仰角 15°, -3dB)/ ベースマネジメント 80Hz / 低域の境界効果 /
    低域の残響を伸ばさない。run.VARIANTS の "v2_tailfix_rear099_wide_array_bm_surr3_elev15_bnd_flatlf" と同一。
    """
    sample_rate: int = TARGET_SR
    hrtf: str = "riec"                  # HRTF_PRESETS のキー、または SOFA のパス
    ear_balance: bool = True

    # 空間
    room_model: str = "v2"              # "v2" or "v1"(1 次反射 + 一様残響)
    rt60_mid: float = RT60_MID
    image_order: int = IMAGE_ORDER
    surround_array: bool = True
    surround_elevation_deg: float = 15.0    # サラウンドの仰角(基準席から見て)。映画館のサラウンドは頭より上
    hrtf_el_interp: bool = True             # HRTF を仰角方向に補間(RIEC は 10° 刻み)
    hrtf_az_interp: bool = False           # 方位角・仰角の両方向に補間(15° 刻みなど粗い格子の HRTF 用)
    hrtf_symmetric: object = False         # True: 左右耳を鏡映しの方向と 1/3oct で揃える / "left"・"right": その耳を鏡映しにして両耳に使う(完全対称)
    late_onset_ms: Optional[tuple] = (12.0, 22.0)  # 残響の立ち上がり (開始, 完全) [ms]。None=平均自由行程→ミキシングタイム
    rear_wall_alpha: Optional[float] = 0.99 # 後壁の吸音率を固定値で上書き(全帯域)
    lf_boundary: bool = True                # 低域の境界効果: スクリーン/サラウンド Q=2(壁)、サブ Q=4(床+前壁)
    lf_rt_factor: float = 1.0               # 31.5Hz の RT60 倍率(Dolby TG 図4.4: 推奨線 1.5 / 下限 1.0)
    front_angle: float = 37.0               # IMAX スクリーン端(ITU-R BS.775 の ±30° より広い)
    timbre_neutral_strength: float = TIMBRE_NEUTRAL_STRENGTH
    cap_early_energy: bool = False          # 初期反射のエネルギーを統計理論の残響予測(Hopkins-Stryker)以内に収める
    neutral_low_hz: float = 40.0            # 音色中立化の下限周波数(これより下は補正を保持)
    tail_lowcut_hz: Optional[float] = None  # 残響ノイズからこれ未満を除く(直流・超低域のばらつき対策)
    sub_room_eq: bool = False               # サブの定常状態の低域の山だけを削る(ルームノードのノッチ, Dolby TG §2.4 / Toole 2015)
    hf_peak_cut: bool = False               # 直接音の 5-16kHz の細い山(非個人 HRTF の耳介共鳴)を 1/3oct の輪郭まで削る。谷は残す(Iida 2007)
    timbre_reference: str = "steady"        # 音色中立化の基準: "steady"=定常状態(SMPTE ST 202), "direct"=直接音(Toole 2015)

    # 低域
    lfe_gain_db: float = LFE_GAIN_DB
    surround_gain_db: float = -3.0      # SL/SR のチャンネルレベル。劇場の 5.1 は -3dB(Dolby TG §2.3.5: 85/82 dBC)
    bass_management: bool = True        # 5ch の低域 + LFE をサブ(スクリーン中央下)から鳴らす
    bass_crossover_hz: float = BASS_CROSSOVER_HZ
    lfe_align: bool = True              # LFE をスクリーン ch の直接音の到達時刻に揃える(劇場のサブ遅延校正相当)
    virtual_bass: bool = False

    # 音色・ヘッドホン
    tonal: str = "mild"                 # "mild" / "thx" / "xcurve" / "off"
    headphone_eq: Optional[str] = None

    # 出力
    target_lufs: float = TARGET_LUFS
    allow_limiting: bool = True             # False: リミッターを使わず、ピークが天井に収まるところまで音量を下げる

    @property
    def hrtf_path(self):
        return HRTF_PRESETS.get(self.hrtf, self.hrtf)


# ---- 部屋モデル v1(2026-09-25 時点の聴感ベスト。再現用に保持) ----
# 1 次 image source(面ごとの 1kHz 反射係数 r)+ 一様 RT の拡散残響(DRR 固定)
V1 = dict(
    rt60=0.30, drr_db=8.0, late_start_ms=20.0, late_hp_hz=150.0, late_lp_hz=6000.0,
    reflection={(0, 0): 0.58, (0, 1): 0.58, (1, 0): 0.30, (1, 1): 0.24, (2, 0): 0.39, (2, 1): 0.24},
)
MOUNTED_WALL_DIST = 1.0      # スピーカーからこの距離未満の壁は「取付面」: その面の反射は作らない
