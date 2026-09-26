"""
Pillar 6: ヘッドホン中和。AutoEq の ParametricEQ.txt 形式を読み、RBJ biquad(SOS)で適用する。

  Preamp: -6.2 dB
  Filter 1: ON LSC Fc 105 Hz Gain 5.5 dB Q 0.70
  Filter 2: ON PK Fc 180 Hz Gain -3.1 dB Q 0.90
"""
import re

import numpy as np
from scipy.signal import sosfilt

from .dsp import rbj_sos

_KIND = {"PK": "peak", "PEQ": "peak", "LSC": "lowshelf", "LS": "lowshelf",
         "HSC": "highshelf", "HS": "highshelf"}
_FILTER = re.compile(
    r"Filter\s*\d*:\s*ON\s+(\w+)\s+Fc\s+([\d.]+)\s*Hz\s+Gain\s+([-\d.]+)\s*dB(?:\s+Q\s+([\d.]+))?", re.I)
_PREAMP = re.compile(r"Preamp:\s*([-\d.]+)\s*dB", re.I)


def load_parametric_eq(path):
    preamp, filters = 0.0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _PREAMP.search(line)
            if m:
                preamp = float(m.group(1))
                continue
            m = _FILTER.search(line)
            if m and m.group(1).upper() in _KIND:
                filters.append((_KIND[m.group(1).upper()], float(m.group(2)),
                                float(m.group(3)), float(m.group(4) or 0.7071)))
    return preamp, filters


def apply_autoeq(stereo, sr, path):
    preamp, filters = load_parametric_eq(path)
    y = stereo * 10 ** (preamp / 20)
    for kind, fc, gain, q in filters:
        y = sosfilt(rbj_sos(kind, fc, gain, q, sr), y, axis=-1)
    return y
