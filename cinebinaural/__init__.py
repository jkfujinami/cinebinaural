"""
cinebinaural — ロスレス 5.1 マスター -> IMAX バイノーラル(プロトタイプ)。

原則: レンダラーは「部屋」と「頭」だけを付け、ミックスは作り直さない。
  - 音色中立 BRIR(ch 毎に両耳平均パワー応答を平坦化、ITD/ILD は保持)
  - IMAX 幾何の image source 初期反射 + 両耳拡散の後期残響
  - LFE +10dB(映画の規約)、無指向
  - ダイナミクス / サチュレーション / 倍音合成は既定 OFF
詳細は THEORY_FOUNDATION.md Part 7。
"""
from .config import PipelineConfig
from .pipeline import process_5p1, render_naive_downmix

__all__ = ["process_5p1", "render_naive_downmix", "PipelineConfig"]
