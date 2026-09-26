"""
cinebinaural 空間音響 (Pillar 2)。

部屋モデル v2(既定):
  BRIR(スピーカー毎) = 直接音 HRIR(t=0, gain=1: 劇場のキャリブレーション相当)
                     + image source 反射(IMAGE_ORDER 次まで)
                         各反射 = HRIR(到来方向) x 面の反射係数 r_s(f)^回数 x 距離減衰
                                  x スピーカー指向性(f, 放射角) x 空気吸収(f, 経路差)
                     + 拡散残響(帯域ごとの RT60(f) で減衰。平均自由行程から立ち上がり、
                       知覚的ミキシングタイムで完全に拡散。量は Hopkins-Stryker の直接/残響比から)
  -> 音色中立化(両耳平均パワー応答を 1/3oct で平坦化、最小位相、両耳同一補正)

  面の反射係数は、素材表の周波数特性を保ったまま、目標 RT60(f) に Eyring 式で一致するよう較正する
  (初期反射と残響の辻褄を合わせるため)。

部屋モデル v1: 1 次反射(1kHz の反射係数)+ 一様 RT の拡散残響。聴感比較の再現用。
部屋モデル none: 直接音のみ。
"""
import numpy as np
import sofar
from scipy.signal import oaconvolve

from . import config as C
from . import dsp


# ---------- スペクトル補助 ----------
def octave_smooth(freqs, mag, frac_oct=1 / 3):
    """1/N オクターブ移動平均(累積和で O(F))。"""
    half = 2 ** (frac_oct / 2)
    lo = np.searchsorted(freqs, freqs / half, side="left")
    hi = np.searchsorted(freqs, freqs * half, side="right")
    cs = np.concatenate([[0.0], np.cumsum(mag)])
    return (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)


def minimum_phase(mag, nfft):
    """振幅応答(rfft 長)から最小位相スペクトルを作る(実ケプストラム法)。"""
    c = np.fft.irfft(np.log(np.maximum(mag, 1e-9)), n=nfft)
    w = np.zeros(nfft)
    w[0] = 1.0
    w[1:nfft // 2] = 2.0
    w[nfft // 2] = 1.0
    return np.exp(np.fft.rfft(c * w, n=nfft))


def log_interp(f, xs, ys):
    """log 周波数上で折れ線補間(範囲外は端の値)。"""
    return np.interp(np.log(np.maximum(f, 1e-3)), np.log(xs), ys)


def band_weights(freqs, centers=C.BAND_CENTERS):
    """隣接する中心周波数の間を log 周波数上の raised-cosine で分けた重み。帯域の合計は常に 1。"""
    lc, lf = np.log2(centers), np.log2(np.maximum(freqs, 1e-3))
    W = np.zeros((len(centers), len(freqs)))
    for b in range(len(centers)):
        w = np.zeros(len(freqs))
        if b == 0:
            w[lf <= lc[0]] = 1.0
        else:
            m = (lf > lc[b - 1]) & (lf <= lc[b])
            w[m] = 0.5 - 0.5 * np.cos(np.pi * (lf[m] - lc[b - 1]) / (lc[b] - lc[b - 1]))
        if b == len(centers) - 1:
            w[lf > lc[-1]] = 1.0
        else:
            m = (lf > lc[b]) & (lf <= lc[b + 1])
            w[m] = 0.5 + 0.5 * np.cos(np.pi * (lf[m] - lc[b]) / (lc[b + 1] - lc[b]))
        W[b] = w
    return W


def air_absorption_db_per_m(f, temp_c=20.0, rh=50.0, p_kpa=101.325):
    """ISO 9613-1 の大気吸収係数 [dB/m]。"""
    T, T0, T01, pr = temp_c + 273.15, 293.15, 273.16, 101.325
    psat = pr * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = rh * psat / p_kpa
    frO = (p_kpa / pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (p_kpa / pr) * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    f = np.asarray(f, dtype=np.float64)
    return 8.686 * f ** 2 * (1.84e-11 * (pr / p_kpa) * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (frO + f ** 2 / frO)
        + 0.1068 * np.exp(-3352.0 / T) / (frN + f ** 2 / frN)))


# ---------- HRTF ----------
def _prepare_foreign_hrir(ir, sr_in, sr_out, pre=48, length=512, lf_flat_hz=200.0):
    """
    48kHz 以外の HRIR を使える形にする(IRCAM LISTEN の無補正 IR を想定)。
      1. 全方向で最も早い到達の pre サンプル前から length サンプル(44.1kHz で約 11.6ms)を切り出し、
         後ろ 1/4 を半ハン窓で落とす(後ろは測定系の響き・雑音)
      2. sr_out へ変換(resample_poly)
      3. lf_flat_hz 未満の振幅を lf_flat_hz の値で平らにする(頭は低域でほぼ透明。測定用スピーカーの低域不足を消す)。
         位相は最小位相 + 元の到達時刻(Kulkarni et al. 1999 のモデル。仰角補間と同じ)
    """
    from fractions import Fraction
    from scipy.signal import resample_poly
    a = np.abs(ir)
    on = (a > 0.2 * a.max(axis=2, keepdims=True)).argmax(axis=2)
    s0 = max(int(on.min()) - pre, 0)
    x = ir[:, :, s0:s0 + length].copy()
    fade = length // 4
    x[:, :, -fade:] *= np.hanning(2 * fade)[fade:]
    r = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    x = resample_poly(x, r.numerator, r.denominator, axis=2)
    N = x.shape[2]
    nfft = 4 * (1 << int(np.ceil(np.log2(N))))
    f = np.fft.rfftfreq(nfft, 1 / sr_out)
    k = np.arange(nfft // 2 + 1)
    out = np.empty_like(x)
    for m in range(x.shape[0]):
        for e in range(2):
            h = x[m, e]
            mag = np.abs(np.fft.rfft(h, n=nfft))
            mag = np.where(f < lf_flat_hz, np.interp(lf_flat_hz, f, octave_smooth(f, mag)), mag)
            delay = HRTF._onset(h)
            out[m, e] = np.fft.irfft(minimum_phase(mag, nfft) * np.exp(-2j * np.pi * k * delay / nfft), n=nfft)[:N]
    return out


class HRTF:
    """SimpleFreeFieldHRIR の SOFA をロードし、方向の最近傍で HRIR を引く。"""

    def __init__(self, sofa_path, target_sr, balance_ears=True):
        try:
            s = sofar.read_sofa(sofa_path, verbose=False)
        except TypeError:
            s = sofar.read_sofa(sofa_path)
        self.ir = np.asarray(s.Data_IR, dtype=np.float64)          # (M, 2, N)
        self.sr = float(s.Data_SamplingRate)
        if int(self.sr) != int(target_sr):
            # 48kHz 以外(IRCAM LISTEN の 44.1kHz など)は、直接音の前後だけ切り出して変換し、低域を平らにする
            self.ir = _prepare_foreign_hrir(self.ir, self.sr, target_sr)
            self.sr = float(target_sr)
        pos = np.asarray(s.SourcePosition, dtype=np.float64)       # az, el, dist
        az, el = np.deg2rad(pos[:, 0]), np.deg2rad(pos[:, 1])
        self._unit = np.ascontiguousarray(np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1))
        self._spec = {}
        self._pos = pos
        self.el_interp = False          # True: 仰角方向に隣接 2 点を補間(最小位相 + 到達時刻)
        self.az_interp = False          # True: 方位角・仰角の両方向に補間(粗い格子の HRTF 用, 最大 4 点)
        self._interp = {}
        if balance_ears:
            self._balance_ears()

    def _balance_ears(self, max_db=10.0):
        """
        全方向平均の左右耳応答を揃える。対称な頭なら全方向平均の L/R は一致するはずで、
        差は測定(マイク感度・耳介の個人差)の偏り。放置すると正面の音が片側に寄る
        (RIEC subject_005 は 8kHz 以上で左耳が 3-8dB 弱い)。方向依存の ILD は変えない。
        """
        N = self.ir.shape[2]
        nfft = 4 * (1 << int(np.ceil(np.log2(N))))
        H = np.fft.rfft(self.ir, n=nfft, axis=2)
        f = np.fft.rfftfreq(nfft, 1 / self.sr)
        df = np.stack([octave_smooth(f, d) for d in np.mean(np.abs(H) ** 2, axis=0)])
        target = df.mean(axis=0)
        for ear in range(2):
            gain_db = np.clip(10 * np.log10(target / df[ear]), -max_db, max_db)
            H[:, ear, :] *= minimum_phase(10 ** (gain_db / 20), nfft)[None, :]
        self.ir = np.ascontiguousarray(np.fft.irfft(H, n=nfft, axis=2)[:, :, :N])

    def mirror_ear(self, master="left"):
        """
        片耳(master)の HRIR を左右反転した方向のもう片耳に使い、振幅・到達時刻・耳介の細部まで完全に左右対称にする。
        RIEC subject_005 は鏡映しの方向どうしで到達時刻が 2-4 サンプル(〜80us)以上ずれ、L/R に同じ音がある
        (ファントム)とき、両耳のくし形の谷の位置がずれる(同じ信号で 1-4kHz に左右 ±6-8dB の差)。
        """
        e = 0 if master == "left" else 1
        mirror = np.argmax(self._unit @ (self._unit * np.array([1.0, -1.0, 1.0])).T, axis=0)
        ir = np.empty_like(self.ir)
        ir[:, e] = self.ir[:, e]
        ir[:, 1 - e] = self.ir[mirror, e]
        self.ir = np.ascontiguousarray(ir)

    def symmetrize(self):
        """
        各方向の左耳を、左右を鏡映しにした方向の右耳と揃える(右耳も同様)。1/3oct の輪郭だけを両者の
        dB 平均に合わせ、細かい構造(耳介の谷)と到達時刻(ITD)は元のまま残す。
        全方向平均で揃える _balance_ears では、特定の方向(正面など)の左右差は残る
        (RIEC subject_005 の正面は 8-12.5kHz で右耳が 4-7dB 強い)。頭は左右対称とみなす。
        """
        N = self.ir.shape[2]
        nfft = 4 * (1 << int(np.ceil(np.log2(N))))
        H = np.fft.rfft(self.ir, n=nfft, axis=2)
        f = np.fft.rfftfreq(nfft, 1 / self.sr)
        env = 10 * np.log10(np.stack([[octave_smooth(f, np.abs(h) ** 2) for h in hm] for hm in H]) + 1e-20)
        mirror = np.argmax(self._unit @ (self._unit * np.array([1.0, -1.0, 1.0])).T, axis=0)
        for i, j in enumerate(mirror):
            for ear in (0, 1):
                gain_db = 0.5 * (env[j, 1 - ear] - env[i, ear])
                H[i, ear] *= minimum_phase(10 ** (gain_db / 20), nfft)
        self.ir = np.ascontiguousarray(np.fft.irfft(H, n=nfft, axis=2)[:, :, :N])

    @property
    def length(self):
        return self.ir.shape[2]

    def nearest_index(self, az_deg, el_deg):
        a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
        v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        return int(np.argmax(self._unit.dot(v)))

    def weights(self, az_deg, el_deg):
        """
        方向 -> [(測定点 index, 重み)]。el_interp=False なら最近傍 1 点。
        True なら、最近傍の方位角の列で、指定仰角を挟む 2 点を仰角の距離で線形補間する
        (同じ方位角では ITD がほぼ同じなので、時間領域の線形補間でコムが出にくい)。
        """
        if self.az_interp:
            return self._weights_grid(az_deg, el_deg)
        i0 = self.nearest_index(az_deg, el_deg)
        if not self.el_interp:
            return [(i0, 1.0)]
        az0, el0 = self._pos[i0, 0], self._pos[i0, 1]
        if abs(el0 - el_deg) < 0.5:
            return [(i0, 1.0)]
        same_az = np.where(np.abs(((self._pos[:, 0] - az0) + 180) % 360 - 180) < 0.5)[0]
        side = same_az[(self._pos[same_az, 1] - el_deg) * (el0 - el_deg) < 0]
        if len(side) == 0:
            return [(i0, 1.0)]
        i1 = side[np.argmin(np.abs(self._pos[side, 1] - el_deg))]
        el1 = self._pos[i1, 1]
        w1 = abs(el_deg - el0) / abs(el1 - el0)
        return [(i0, 1.0 - w1), (int(i1), w1)]

    def _weights_grid(self, az_deg, el_deg):
        """仰角で挟む 2 行 × 各行で方位角を挟む 2 点の双線形の重み(格子上ならその点だけ)。"""
        az = az_deg % 360.0
        els = np.unique(np.round(self._pos[:, 1], 3))
        lo, hi = els[els <= el_deg + 1e-6], els[els >= el_deg - 1e-6]
        rows = []
        if len(lo) == 0 or len(hi) == 0:
            rows = [(els[np.argmin(np.abs(els - el_deg))], 1.0)]
        elif abs(lo[-1] - hi[0]) < 1e-6:
            rows = [(lo[-1], 1.0)]
        else:
            w = (el_deg - lo[-1]) / (hi[0] - lo[-1])
            rows = [(lo[-1], 1.0 - w), (hi[0], w)]
        out = []
        for e, we in rows:
            idx = np.where(np.abs(np.round(self._pos[:, 1], 3) - e) < 1e-6)[0]
            a = self._pos[idx, 0] % 360.0
            if len(idx) == 1:
                out.append((int(idx[0]), we))
                continue
            d = (a - az) % 360.0                     # 反時計回りに見た差
            j1 = idx[np.argmin(d)]                   # az 以上で最も近い
            j0 = idx[np.argmax(d)]                   # az 以下で最も近い
            d1 = (self._pos[j1, 0] - az) % 360.0
            d0 = (az - self._pos[j0, 0]) % 360.0
            if d1 < 1e-6 or d0 + d1 < 1e-6:
                out.append((int(j1), we))
            else:
                out += [(int(j0), we * d1 / (d0 + d1)), (int(j1), we * d0 / (d0 + d1))]
        return [(i, w) for i, w in out if w > 1e-9]

    def _interp_multi(self, w):
        """複数点の HRIR を「振幅(最小位相)+ 到達時刻」で重み付き補間(_interp_ir の多点版)。"""
        key = tuple((i, round(x, 6)) for i, x in w)
        if key not in self._interp:
            N = self.length
            nfft = 4 * (1 << int(np.ceil(np.log2(N))))
            f = np.arange(nfft // 2 + 1)
            out = np.zeros((2, N))
            for ear in range(2):
                mag = sum(x * np.abs(np.fft.rfft(self.ir[i, ear], n=nfft)) for i, x in w)
                delay = sum(x * self._onset(self.ir[i, ear]) for i, x in w)
                H = minimum_phase(mag, nfft) * np.exp(-2j * np.pi * f * delay / nfft)
                out[ear] = np.fft.irfft(H, n=nfft)[:N]
            self._interp[key] = out
        return self._interp[key]

    @staticmethod
    def _onset(h, rel=0.2):
        """到達時刻 [samples](|h| が最大値の rel 倍を初めて超える点、線形補間でサンプル以下まで)。"""
        a = np.abs(h)
        thr = rel * a.max()
        k = int(np.argmax(a >= thr))
        if k == 0:
            return 0.0
        return k - 1 + (thr - a[k - 1]) / max(a[k] - a[k - 1], 1e-30)

    def _interp_ir(self, i0, i1, w1):
        """
        2 点の HRIR を「振幅特性(最小位相)+ 到達時刻」に分けて別々に補間する。
        時間波形のまま平均すると、サンプル以下の到達ずれで高域が打ち消し合う(実測 5-12kHz で -6.7dB)。
        最小位相 + 遅延のモデルは聴感上十分とされる(Kulkarni et al. 1999)。
        """
        key = (i0, i1, round(w1, 4))
        if key not in self._interp:
            N = self.length
            nfft = 4 * (1 << int(np.ceil(np.log2(N))))
            f = np.arange(nfft // 2 + 1)
            out = np.zeros((2, N))
            for ear in range(2):
                h0, h1 = self.ir[i0, ear], self.ir[i1, ear]
                mag = ((1 - w1) * np.abs(np.fft.rfft(h0, n=nfft)) + w1 * np.abs(np.fft.rfft(h1, n=nfft)))
                delay = (1 - w1) * self._onset(h0) + w1 * self._onset(h1)
                H = minimum_phase(mag, nfft) * np.exp(-2j * np.pi * f * delay / nfft)
                out[ear] = np.fft.irfft(H, n=nfft)[:N]
            self._interp[key] = out
        return self._interp[key]

    def nearest(self, az_deg, el_deg):
        w = self.weights(az_deg, el_deg)
        if len(w) == 1:
            return self.ir[w[0][0]]
        if self.az_interp:
            return self._interp_multi(w)
        return self._interp_ir(w[0][0], w[1][0], w[1][1])

    def spectrum_dir(self, az_deg, el_deg, nfft):
        w = self.weights(az_deg, el_deg)
        if len(w) == 1:
            return self.spectrum(w[0][0], nfft)
        if self.az_interp:
            key = ("multi", tuple((i, round(x, 6)) for i, x in w), nfft)
            if key not in self._spec:
                self._spec[key] = np.fft.rfft(self._interp_multi(w), n=nfft, axis=1)
            return self._spec[key]
        key = ("interp", w[0][0], w[1][0], round(w[1][1], 4), nfft)
        if key not in self._spec:
            self._spec[key] = np.fft.rfft(self._interp_ir(w[0][0], w[1][0], w[1][1]), n=nfft, axis=1)
        return self._spec[key]

    def spectrum(self, idx, nfft):
        key = (idx, nfft)
        if key not in self._spec:
            self._spec[key] = np.fft.rfft(self.ir[idx], n=nfft, axis=1)
        return self._spec[key]


# ---------- 幾何 ----------
_FORWARD = np.array([0.0, -1.0, 0.0])   # スクリーン(y=0)を向く
_LEFT = np.array([1.0, 0.0, 0.0])
_UP = np.array([0.0, 0.0, 1.0])


def load_hrtf(cfg, sr):
    """設定どおりに HRTF を読み込む(左右補正、左右対称化、仰角補間)。"""
    hrtf = HRTF(cfg.hrtf_path, sr, balance_ears=cfg.ear_balance)
    sym = getattr(cfg, "hrtf_symmetric", False)
    if sym in ("left", "right"):
        hrtf.mirror_ear(sym)
    elif sym:
        hrtf.symmetrize()
    hrtf.el_interp = cfg.hrtf_el_interp
    hrtf.az_interp = getattr(cfg, "hrtf_az_interp", False)
    return hrtf


def direction_to_vector(az_deg, el_deg):
    a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.cos(e) * (np.cos(a) * _FORWARD + np.sin(a) * _LEFT) + np.sin(e) * _UP


def vector_to_direction(v):
    f, l, u = v @ _FORWARD, v @ _LEFT, v @ _UP
    return np.degrees(np.arctan2(l, f)) % 360.0, np.degrees(np.arctan2(u, np.hypot(f, l)))


def channel_directions(cfg):
    fa, sa = cfg.front_angle, C.SURROUND_ANGLE
    se = cfg.surround_elevation_deg
    return {"FL": (fa, 0.0), "FR": (360.0 - fa, 0.0), "FC": (0.0, 0.0),
            "SL": (sa, se), "SR": (360.0 - sa, se)}


def speaker_position(az_deg, el_deg):
    """聴取位置から (az, el) 方向へ伸ばし、最初に当たる壁の手前に置く。"""
    L, dims, m = np.array(C.LISTENER_POS), np.array(C.ROOM_DIMS), C.SPEAKER_WALL_MARGIN
    d = direction_to_vector(az_deg, el_deg)
    ts = []
    for k in range(3):
        if d[k] > 1e-9:
            ts.append((dims[k] - m - L[k]) / d[k])
        elif d[k] < -1e-9:
            ts.append((m - L[k]) / d[k])
    return L + min(ts) * d


def surround_array_positions(side, n=C.SURROUND_ARRAY_N, elevation_deg=0.0):
    """
    Dolby TG §2.3.1: 側壁の中ほどから後壁(の中央まで)に均等配置。side=+1 左, -1 右。
    高さは聴取位置と同じ(5.1 の水平レイアウトを保つ)。
    """
    W, D, _ = C.ROOM_DIMS
    m, z = C.SPEAKER_WALL_MARGIN, C.LISTENER_POS[2]
    x_wall = W - m if side > 0 else m
    seg1, seg2 = (D - m) - D / 2, abs(x_wall - W / 2)
    L = np.array(C.LISTENER_POS)
    pts = []
    for k in range(n):
        s = (k + 0.5) / n * (seg1 + seg2)
        if s <= seg1:
            p = np.array([x_wall, D / 2 + s, z])
        else:
            p = np.array([x_wall - side * (s - seg1), D - m, z])
        # 基準席から見た仰角が elevation_deg になる高さに付ける
        p[2] = L[2] + np.hypot(*(p[:2] - L[:2])) * np.tan(np.deg2rad(elevation_deg))
        pts.append(p)
    return pts


def mounted_surface(spk):
    """スピーカーの取付面(壁から MOUNTED_WALL_DIST 未満)。その面の反射は作らない。"""
    for axis in range(3):
        for side, wall in ((0, 0.0), (1, C.ROOM_DIMS[axis])):
            if abs(spk[axis] - wall) < C.MOUNTED_WALL_DIST:
                return (axis, side)
    return None


def image_sources(spk, max_order):
    """
    直方体室の image source (Allen & Berkley)。各軸 x_img = (1-2p) x_s + 2 n L。
    座標 0 側の壁に |n-p| 回、L 側の壁に |n| 回反射する。p=1 の軸は放射方向が反転する。
    戻り値: (img, {(axis, side): 回数}, flip(3,), 次数)
    """
    per_axis = []
    for axis in range(3):
        Lx, s = C.ROOM_DIMS[axis], spk[axis]
        opts = []
        for n in range(-max_order, max_order + 1):
            for p in (0, 1):
                c0, c1 = abs(n - p), abs(n)
                if c0 + c1 <= max_order:
                    opts.append(((1 - 2 * p) * s + 2 * n * Lx, c0, c1, 1 - 2 * p))
        per_axis.append(opts)
    for ox in per_axis[0]:
        for oy in per_axis[1]:
            for oz in per_axis[2]:
                order = ox[1] + ox[2] + oy[1] + oy[2] + oz[1] + oz[2]
                if order > max_order:
                    continue
                img = np.array([ox[0], oy[0], oz[0]])
                counts = {(0, 0): ox[1], (0, 1): ox[2], (1, 0): oy[1],
                          (1, 1): oy[2], (2, 0): oz[1], (2, 1): oz[2]}
                yield img, counts, np.array([ox[3], oy[3], oz[3]], dtype=float), order


# ---------- 部屋の音響(v2) ----------
class RoomAcoustics:
    """面の吸音を、素材の周波数特性を保ったまま目標 RT60(f) に Eyring 式で較正する。"""

    def __init__(self, rt60_mid, alpha_override=None, lf_rt_factor=None):
        self.alpha_override = alpha_override or {}
        self.rt_factor = list(C.RT_FREQ_FACTOR)
        if lf_rt_factor is not None:
            self.rt_factor[0] = (self.rt_factor[0][0], lf_rt_factor)
        W, D, H = C.ROOM_DIMS
        self.V = W * D * H
        self.area = {(0, 0): D * H, (0, 1): D * H, (1, 0): W * H,
                     (1, 1): W * H, (2, 0): W * D, (2, 1): W * D}
        self.S = sum(self.area.values())
        self.rt60_mid = rt60_mid
        self.t_mfp = 4 * self.V / self.S / C.SPEED_OF_SOUND        # 平均自由行程の時間
        self.t_mix = (0.0117 * self.V + 50.1) / 1000.0            # Lindau 2012: t_mp95 [s]

    def target_rt(self, f):
        fx, fy = zip(*self.rt_factor)
        return self.rt60_mid * log_interp(f, fx, fy)

    def _material_alpha(self, key, f):
        _, mix = C.SURFACES[key]
        a = sum(frac * log_interp(f, C.MATERIAL_FREQS, C.MATERIALS[mat]) for mat, frac in mix.items())
        return np.clip(a, 0.01, 0.99)

    def calibrated_alpha(self, f):
        """{面: α'(f)}, 平均 α'(f), 空気の強度減衰係数 m(f) [1/m]。α' = 1 - (1-α)^k(f)。"""
        m = air_absorption_db_per_m(f) / (10 * np.log10(np.e))
        need = 0.161 * self.V / self.target_rt(f) - 4 * m * self.V
        base = sum(self.area[k] * -np.log(1 - self._material_alpha(k, f)) for k in self.area)
        kf = np.clip(need / base, 0.05, 50.0)
        alpha = {k: 1 - (1 - self._material_alpha(k, f)) ** kf for k in self.area}
        mean = sum(self.area[k] * alpha[k] for k in self.area) / self.S
        for k, a in self.alpha_override.items():      # 反射(image source)にだけ効く上書き
            alpha[k] = np.full_like(np.asarray(f, dtype=float), a)
        return alpha, np.clip(mean, 1e-3, 0.999), m

    def reverb_to_direct(self, f, q, d):
        """Hopkins-Stryker: 残響/直接のエネルギー比 = 16π d^2 / (Q R), R = Sᾱ/(1-ᾱ) + 4mV。"""
        _, mean, m = self.calibrated_alpha(f)
        R = self.S * mean / (1 - mean) + 4 * m * self.V
        return 16 * np.pi * d ** 2 / (q * R)


def directivity_weight(f):
    """低域 0(無指向)→ 高域 1(フル指向性)。DIRECTIVITY_FREQS の間を log で補間。"""
    f0, f1 = C.DIRECTIVITY_FREQS
    return np.clip(np.log(np.maximum(f, 1e-3) / f0) / np.log(f1 / f0), 0.0, 1.0)


def directivity_gain(f, aim, emit, beam):
    """楕円ビーム(-6dB 片側角 beam=(h, v))。低域は無指向。振幅ゲインを返す。"""
    a = aim / np.linalg.norm(aim)
    r = np.cross(a, _UP)
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-9 else _LEFT
    u = np.cross(r, a)
    e = emit / np.linalg.norm(emit)
    th = np.degrees(np.arctan2(e @ r, e @ a))
    tv = np.degrees(np.arctan2(e @ u, np.hypot(e @ a, e @ r)))
    db = max(-6.0 * ((th / beam[0]) ** 2 + (tv / beam[1]) ** 2), C.DIRECTIVITY_FLOOR_DB)
    return 10 ** (db * directivity_weight(f) / 20)


def beam_q(f, beam, q_lf=1.0):
    """
    指向係数 Q(f)。高域は -6dB ビーム幅から DI ≈ 10log(41253 / (2h * 2v))。
    低域は無指向だが、壁や床に接していれば境界効果で q_lf(半空間 2, 1/4 空間 4)になる。
    """
    q_hf = max(41253.0 / (2 * beam[0] * 2 * beam[1]), q_lf, 1.0)
    return q_lf + (q_hf - q_lf) * directivity_weight(f)


def speaker_response(hrtf, room, spk, beam, sr, nfft, freqs, max_order):
    """1 本のスピーカーの (直接音スペクトル, 反射スペクトル, 距離, 反射数)。両耳 (2, F)。"""
    L = np.array(C.LISTENER_POS)
    d0 = np.linalg.norm(spk - L)
    aim = L - spk
    B_dir = np.array(hrtf.spectrum_dir(*vector_to_direction(spk - L), nfft), copy=True)
    B_ref = np.zeros_like(B_dir)
    alpha, _, _ = room.calibrated_alpha(freqs)
    refl = {k: np.sqrt(1 - alpha[k]) for k in alpha}
    mounted = mounted_surface(spk)
    if mounted is not None:
        refl[mounted] = np.zeros_like(freqs)
    air = air_absorption_db_per_m(freqs)
    count = 0
    for img, counts, flip, order in image_sources(spk, max_order):
        if order == 0 or (mounted is not None and counts[mounted] > 0):
            continue
        v = img - L
        di = np.linalg.norm(v)
        g = d0 / di * 10 ** (-air * (di - d0) / 20) * directivity_gain(freqs, aim, (L - img) * flip, beam)
        for k, c in counts.items():
            if c:
                g = g * refl[k] ** c
        tau = (di - d0) / C.SPEED_OF_SOUND
        H = hrtf.spectrum_dir(*vector_to_direction(v), nfft)
        B_ref += g[None, :] * H * np.exp(-2j * np.pi * freqs * tau)[None, :]
        count += 1
    return B_dir, B_ref, d0, count


def late_tail_bands(sr, n, rt_bands, t_start, t_full, seed, low_cut_hz=None):
    """
    帯域ごとの拡散残響成分(各 (2, n)、エネルギー 1 に正規化)。
    低域は両耳コヒーレント。立ち上がりは t_start から t_full まで raised-cosine。
    """
    rng = np.random.default_rng(seed)
    xo = C.LATE_COHERENT_BELOW_HZ
    common = dsp.lowpass(rng.standard_normal(n), xo, sr)
    ears = np.stack([common + dsp.highpass(rng.standard_normal(n), xo, sr) for _ in range(2)])
    F = np.fft.rfft(ears, axis=1)
    fr = np.fft.rfftfreq(n, 1 / sr)
    if low_cut_hz:
        F[:, fr < low_cut_hz] = 0.0
    W = band_weights(fr)
    t = np.arange(n) / sr
    x = np.clip((t - t_start) / max(t_full - t_start, 1e-3), 0.0, 1.0)
    onset = 0.5 - 0.5 * np.cos(np.pi * x)
    comps = []
    for b, T in enumerate(rt_bands):
        c = np.fft.irfft(F * W[b][None, :], n=n, axis=1) * (np.exp(-6.9078 * t / T) * onset)[None, :]
        comps.append(c / np.sqrt(np.mean(np.sum(c ** 2, axis=1)) + 1e-30))
    return comps


def _band_energy(Wb, B, nfft):
    """
    rfft スペクトル (2, F) の帯域別エネルギーを、時間領域のエネルギー(Σx^2)の単位で返す(両耳平均)。
    Parseval: Σx^2 = (|X_0|^2 + 2Σ|X_k|^2 + |X_{N/2}|^2) / N。残響成分は時間領域で正規化しているので単位を揃える。
    """
    p = np.mean(np.abs(B) ** 2, axis=0)
    p[1:-1] *= 2.0
    return (Wb * p[None, :]).sum(axis=1) / nfft


def make_brir_v2(hrtf, room, speakers, beam, sr, cfg, seed, info, q_lf=1.0):
    """
    speakers: [(位置, 振幅ゲイン, 遅延秒)]。点音源は 1 本、アレイは複数。
    直接音 + image source 反射を周波数領域で合成し、帯域ごとの残響を足す。
    """
    L = np.array(C.LISTENER_POS)
    max_tau = 0.0
    for spk, _, delay in speakers:
        d0 = np.linalg.norm(spk - L)
        for img, _, _, order in image_sources(spk, cfg.image_order):
            max_tau = max(max_tau, (np.linalg.norm(img - L) - d0) / C.SPEED_OF_SOUND + delay)
    nfft = 1 << int(np.ceil(np.log2(int(max_tau * sr) + hrtf.length + 1024)))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    Wb = band_weights(freqs)
    centers = np.array(C.BAND_CENTERS)

    B = np.zeros((2, len(freqs)), dtype=complex)
    B_refl = np.zeros_like(B)
    e_dir = np.zeros(len(centers))
    e_rev = np.zeros(len(centers))
    n_refl = 0
    for spk, gain, delay in speakers:
        B_d, B_r, d0, cnt = speaker_response(hrtf, room, spk, beam, sr, nfft, freqs, cfg.image_order)
        n_refl += cnt
        shift = gain * np.exp(-2j * np.pi * freqs * delay)[None, :]
        B += (B_d + B_r) * shift
        B_refl += B_r * shift
        e_d = _band_energy(Wb, B_d, nfft) * gain ** 2
        e_dir += e_d
        e_rev += e_d * room.reverb_to_direct(centers, beam_q(centers, beam, q_lf), d0)
    e_ref = _band_energy(Wb, B_refl, nfft)
    capped = []
    if getattr(cfg, "cap_early_energy", False):
        # 初期反射が統計理論の残響エネルギー予測(e_rev)の 9 割を超える帯域では、反射を下げて予測内に収める。
        # 超えた分がそのまま上乗せされていた(サブの 80Hz 以下が 2-4dB 重くなっていた原因)。
        g = np.sqrt(np.minimum(1.0, 0.9 * e_rev / np.maximum(e_ref, 1e-30)))
        capped = [(c, 20 * np.log10(v)) for c, v in zip(centers, g) if v < 0.999]
        G = (Wb * g[:, None]).sum(axis=0)[None, :]
        B = B - B_refl + B_refl * G
        B_refl = B_refl * G
        e_ref = _band_energy(Wb, B_refl, nfft)
    e_tail = np.maximum(e_rev - e_ref, 0.1 * e_rev)

    early = np.fft.irfft(B, n=nfft, axis=1)
    rt_bands = room.target_rt(centers)
    n_tail = int(sr * (room.t_mix + 1.2 * rt_bands.max()))
    comps = late_tail_bands(sr, n_tail, rt_bands, room.t_mfp, room.t_mix, seed,
                            low_cut_hz=getattr(cfg, "tail_lowcut_hz", None))
    tail = sum(np.sqrt(e) * c for e, c in zip(e_tail, comps))

    brir = np.zeros((2, max(nfft, n_tail)))
    brir[:, :nfft] += early
    brir[:, :n_tail] += tail
    if info is not None:
        k1 = int(np.argmin(np.abs(centers - 1000)))
        k63 = int(np.argmin(np.abs(centers - 63)))
        info.update(n_refl=n_refl, capped=capped, drr_1k=10 * np.log10(e_dir[k1] / e_rev[k1]),
                    drr_63=10 * np.log10(e_dir[k63] / e_rev[k63]),
                    early_share_1k=min(e_ref[k1] / e_rev[k1], 1.0))
    return brir


# ---------- 部屋モデル v1(再現用) ----------
def _late_tail_v1(sr, n, rt60, seed):
    rng = np.random.default_rng(seed)
    xo = C.LATE_COHERENT_BELOW_HZ
    common = dsp.lowpass(rng.standard_normal(n), xo, sr)
    ears = [common + dsp.highpass(rng.standard_normal(n), xo, sr) for _ in range(2)]
    tail = np.stack([dsp.bandpass(e, C.V1["late_hp_hz"], C.V1["late_lp_hz"], sr, order=2) for e in ears])
    t = np.arange(n) / sr
    tail *= np.exp(-6.9078 * t / rt60) * np.clip((t - C.V1["late_start_ms"] / 1000.0) / 0.010, 0.0, 1.0)
    return tail / np.sqrt(np.mean(np.sum(tail ** 2, axis=1)) + 1e-20)


def make_brir_v1(hrtf, spk, sr, seed):
    L = np.array(C.LISTENER_POS)
    d0 = np.linalg.norm(spk - L)
    N = hrtf.length
    rt60 = C.V1["rt60"]
    n = int(sr * (1.5 * rt60 + 0.15))
    brir = np.zeros((2, n + N))
    h = hrtf.nearest(*vector_to_direction(spk - L))
    brir[:, :N] += h
    mounted = mounted_surface(spk)
    for img, counts, _, order in image_sources(spk, 1):
        if order != 1:
            continue
        key = next(k for k, c in counts.items() if c)
        if key == mounted:
            continue
        v = img - L
        di = np.linalg.norm(v)
        k = int(round((di - d0) / C.SPEED_OF_SOUND * sr))
        if k + N <= brir.shape[1]:
            brir[:, k:k + N] += C.V1["reflection"][key] * d0 / di * hrtf.nearest(*vector_to_direction(v))
    e_direct = np.mean(np.sum(h ** 2, axis=1))
    brir[:, :n] += _late_tail_v1(sr, n, rt60, seed) * np.sqrt(e_direct * 10 ** (-C.V1["drr_db"] / 10))
    return brir


# ---------- 音色中立化 ----------
def direct_only_brir(hrtf, speakers, sr):
    """直接音だけの BRIR(各スピーカーの HRIR をゲイン・遅延付きで足したもの)。反射も残響も含まない。"""
    L = np.array(C.LISTENER_POS)
    N = hrtf.length
    n = N + int(max(d for _, _, d in speakers) * sr) + 1
    out = np.zeros((2, n))
    for spk, gain, delay in speakers:
        k = int(round(delay * sr))
        out[:, k:k + N] += gain * hrtf.nearest(*vector_to_direction(spk - L))
    return out


def neutralize_timbre(brir, sr, strength=1.0, smooth_oct=1 / 3, max_db=15.0, reference=None, low_hz=40.0):
    """
    両耳平均パワー応答を 1/N oct 平滑化して平坦化する。補正は最小位相で両耳同一
    -> ITD / ILD / 平滑化幅より細かい構造(部屋のコム、耳介ノッチの細部)は保たれ、
    方向に依らない色付け(声質を変える主因)だけが消える。
    """
    n = brir.shape[1]
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    B = np.fft.rfft(brir, n=nfft, axis=1)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    # reference を渡すと、その応答(例: 直接音だけ)を平坦にする補正を brir 全体に掛ける
    R = B if reference is None else np.fft.rfft(reference, n=nfft, axis=1)
    p = 0.5 * (np.abs(R[0]) ** 2 + np.abs(R[1]) ** 2)
    corr_db = -10 * np.log10(octave_smooth(freqs, p, smooth_oct) + 1e-20) * strength
    band = (freqs >= low_hz) & (freqs <= 16000)         # 帯域外は端の値で保持(過大ブースト防止)
    corr_db[freqs < low_hz] = corr_db[band][0]
    corr_db[freqs > 16000] = corr_db[band][-1]
    corr_db = np.clip(corr_db, -max_db, max_db)
    H = minimum_phase(10 ** (corr_db / 20), nfft)
    return np.fft.irfft(B * H[None, :], n=nfft, axis=1)[:, : n + 4096]


def cut_pinna_peaks(brir, reference, sr, band=None, fine_oct=None, env_oct=None):
    """
    直接音(reference)の各耳の、1/fine_oct で見た山のうち 1/env_oct の輪郭から突き出た分だけを削る。
    対象は band(既定 5-16kHz)。谷は残す。山の分だけ 1/3oct の輪郭も下がる(実測で帯域平均 0.2-1.0dB)。
    上下の定位は N1・N2・P1(約 4kHz)でほぼ決まる(Iida et al. 2007)ので、P1 より上の細い山は
    方向の手がかりというより非個人 HRTF の色付けになる。補正は耳ごとの最小位相。
    """
    band = band or C.HF_PEAK_CUT_HZ
    fine_oct = fine_oct or C.HF_PEAK_FINE_OCT
    env_oct = env_oct or C.TIMBRE_SMOOTH_OCT
    n = brir.shape[1]
    nfft = 1 << (int(np.ceil(np.log2(max(n, reference.shape[1])))) + 1)
    B = np.fft.rfft(brir, n=nfft, axis=1)
    R = np.fft.rfft(reference, n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    lo, hi = band
    w = np.clip(np.log2(f / (lo / 2 ** 0.25) + 1e-12) / 0.25, 0, 1) * np.clip(np.log2(hi * 2 ** 0.25 / (f + 1e-12)) / 0.25, 0, 1)
    out = np.empty_like(B)
    for ear in (0, 1):
        p = np.abs(R[ear]) ** 2
        fine = 10 * np.log10(octave_smooth(f, p, fine_oct) + 1e-20)
        env = 10 * np.log10(octave_smooth(f, p, env_oct) + 1e-20)
        cut = np.minimum(0.0, env - fine) * w
        out[ear] = B[ear] * minimum_phase(10 ** (cut / 20), nfft)
    return np.fft.irfft(out, n=nfft, axis=1)[:, : n + 4096]


# ---------- チャンネル BRIR の構築 ----------
def _channel_speakers(name, az, el, cfg):
    """チャンネル -> [(位置, 振幅ゲイン, 遅延秒)]。"""
    if cfg.surround_array and name in C.SURROUND_CHANNELS:
        L = np.array(C.LISTENER_POS)
        pts = surround_array_positions(+1 if name == "SL" else -1, elevation_deg=cfg.surround_elevation_deg)
        d = np.array([np.linalg.norm(p - L) for p in pts])
        gains = (d.mean() / d) / np.sqrt(len(pts))           # アレイ全体で 1 本分の音圧に校正
        delays = (d - d.min()) / C.SPEED_OF_SOUND            # 距離差による自然な到達時間差
        return list(zip(pts, gains, delays))
    return [(speaker_position(az, el), 1.0, 0.0)]


def _make_room(cfg):
    room = RoomAcoustics(cfg.rt60_mid,
                         {(1, 1): cfg.rear_wall_alpha} if cfg.rear_wall_alpha is not None else None,
                         cfg.lf_rt_factor)
    if cfg.late_onset_ms is not None:
        room.t_mfp, room.t_mix = (x / 1000.0 for x in cfg.late_onset_ms)
    return room


def build_brirs(hrtf, sr, cfg, log=None):
    room = _make_room(cfg)
    if log and cfg.room_model == "v2":
        c = np.array(C.BAND_CENTERS)
        log(f"    room v2: V={room.V:.0f}m3, 平均自由行程 {room.t_mfp * 1000:.0f}ms, "
            f"ミキシングタイム {room.t_mix * 1000:.0f}ms, image 次数 {cfg.image_order}")
        log("    RT60(f): " + " ".join(f"{f:g}:{t:.2f}" for f, t in zip(c, room.target_rt(c))))
    brirs = {}
    for i, (name, (az, el)) in enumerate(channel_directions(cfg).items()):
        speakers = _channel_speakers(name, az, el, cfg)
        info = {}
        if cfg.room_model == "v2":
            beam = C.SCREEN_BEAM if name in C.SCREEN_CHANNELS else C.SURROUND_BEAM
            q_lf = (C.LF_BOUNDARY_Q["screen" if name in C.SCREEN_CHANNELS else "surround"]
                    if cfg.lf_boundary else 1.0)
            raw = make_brir_v2(hrtf, room, speakers, beam, sr, cfg, seed=i + 1, info=info, q_lf=q_lf)
        elif cfg.room_model == "v1":
            raw = make_brir_v1(hrtf, speakers[0][0], sr, seed=i + 1)
        else:   # none: 直接音のみ
            L = np.array(C.LISTENER_POS)
            raw = hrtf.nearest(*vector_to_direction(speakers[0][0] - L)).copy()
        ref = direct_only_brir(hrtf, speakers, sr) if cfg.timbre_reference == "direct" else None
        brirs[name] = (neutralize_timbre(raw, sr, cfg.timbre_neutral_strength, C.TIMBRE_SMOOTH_OCT,
                                         reference=ref, low_hz=cfg.neutral_low_hz)
                       if cfg.timbre_neutral_strength > 0 else raw)
        if getattr(cfg, "hf_peak_cut", False):
            brirs[name] = cut_pinna_peaks(brirs[name], direct_only_brir(hrtf, speakers, sr), sr)
        if log and info:
            if info.get("capped"):
                log(f"    {name}: 初期反射を抑えた帯域 " + " ".join(f"{c:g}Hz:{d:+.1f}dB" for c, d in info["capped"]))
            log(f"    {name}: スピーカー {len(speakers)} 本, 反射 {info['n_refl']}, "
                f"DRR@63Hz {info['drr_63']:+.1f}dB / @1k {info['drr_1k']:+.1f}dB, "
                f"残響のうち初期反射 {info['early_share_1k'] * 100:.0f}%")
    return brirs


def notch_room_peaks(brir, sr, f_max=120.0, f_min=20.0):
    """
    定常状態の低域の山だけを削る(谷は埋めない)。1/3oct で平滑化したパワーの、f_min-f_max 帯の中央値を
    基準に、それを超える分だけ下げる。劇場のルームノードのノッチ補正(Dolby TG §2.4.4)に相当。
    戻り: (補正後 BRIR, [(周波数, 削った dB)])
    """
    n = brir.shape[1]
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    B = np.fft.rfft(brir, n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    p_db = 10 * np.log10(octave_smooth(f, 0.5 * (np.abs(B[0]) ** 2 + np.abs(B[1]) ** 2)) + 1e-30)
    band = (f >= f_min) & (f <= f_max)
    ref = np.median(p_db[band])
    cut = np.where(band, np.minimum(0.0, ref - p_db), 0.0)
    taper = (f > f_max) & (f < f_max * 1.33)                 # 上端はなめらかに 0 へ
    cut[taper] = cut[band][-1] * (1 - (f[taper] - f_max) / (f_max * 0.33))
    H = minimum_phase(10 ** (cut / 20), nfft)
    out = np.fft.irfft(B * H[None, :], n=nfft, axis=1)[:, : n + 4096]
    marks = [(c, float(np.interp(c, f, cut))) for c in (20, 25, 31.5, 40, 50, 63, 80, 100)]
    return out, [(c, d) for c, d in marks if d < -0.3]


def build_sub_brir(hrtf, sr, cfg, log=None):
    """サブウーファー(スクリーン中央下)の BRIR。低域は無指向なのでビームは使わない。"""
    spk = np.array(C.SUB_POS_ON_FLOOR if cfg.lf_boundary else C.SUB_POS, dtype=float)
    L = np.array(C.LISTENER_POS)
    info = {}
    if cfg.room_model == "v2":
        q_lf = C.LF_BOUNDARY_Q["sub"] if cfg.lf_boundary else 1.0
        raw = make_brir_v2(hrtf, _make_room(cfg), [(spk, 1.0, 0.0)], (1e6, 1e6), sr, cfg,
                           seed=99, info=info, q_lf=q_lf)
    elif cfg.room_model == "v1":
        raw = make_brir_v1(hrtf, spk, sr, seed=99)
    else:
        raw = hrtf.nearest(*vector_to_direction(spk - L)).copy()
    if log:
        az, el = vector_to_direction(spk - L)
        drr = f", DRR@63Hz {info['drr_63']:+.1f}dB" if info else ""
        if info.get("capped"):
            drr += ", 初期反射を抑えた帯域 " + " ".join(f"{c:g}Hz:{d:+.1f}dB" for c, d in info["capped"])
        log(f"    SUB: 位置 ({spk[0]:g}, {spk[1]:g}, {spk[2]:g}), 方位 {az:.0f}°, 仰角 {el:.1f}°, "
            f"距離 {np.linalg.norm(spk - L):.1f}m{drr}")
    ref = direct_only_brir(hrtf, [(spk, 1.0, 0.0)], sr) if cfg.timbre_reference == "direct" else None
    out = (neutralize_timbre(raw, sr, cfg.timbre_neutral_strength, C.TIMBRE_SMOOTH_OCT, reference=ref,
                             low_hz=cfg.neutral_low_hz)
           if cfg.timbre_neutral_strength > 0 else raw)
    if getattr(cfg, "sub_room_eq", False):
        out, cuts = notch_room_peaks(out, sr)
        if log:
            log("    SUB: ルームノードのノッチ " + (" ".join(f"{c:g}Hz:{d:+.1f}dB" for c, d in cuts) or "なし"))
    return out


def direct_arrival_samples(brirs, names=C.SCREEN_CHANNELS):
    """
    スクリーン ch の直接音の到達時刻 [samples](両耳平均の広帯域ピークの中央値)。
    HRIR 自体が持つ到達遅延(RIEC で約 2.5ms)を含む。LFE をこれに揃える。
    """
    return int(np.median([np.argmax(np.abs(brirs[n].mean(axis=0))) for n in names]))


def render_binaural(directional, lfe, brirs):
    """directional: name -> mono。lfe: mono(無指向、両耳へ同相)または None。戻り (2, N)。"""
    n = max(len(v) for v in directional.values())
    m = max(b.shape[1] for b in brirs.values())
    out = np.zeros((2, n + m - 1))
    for name, sig in directional.items():
        b = brirs[name]
        for ear in (0, 1):
            y = oaconvolve(sig, b[ear])
            out[ear, : len(y)] += y
    if lfe is not None:
        out[:, : len(lfe)] += lfe[None, :]
    return out
