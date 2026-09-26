


# cinebinaural

**IMAX-like cinema sound on your headphones.**
An open, physically grounded binaural virtualizer, in the spirit of Dolby Atmos for Headphones. It renders 5.1 audio as if you were sitting in a virtual premium large-format (IMAX-like) theater. It also includes an AI upmixer that turns stereo music into 5.1 first.

It virtualizes 5.1 channel-based audio. It does not decode or render Dolby Atmos object/height content.

Every parameter comes from a paper, a standard or a measurement: the room acoustics, the speaker layout, the calibration and the HRTF processing. Each design change was A/B-tested by ear at matched loudness (LUFS).

> Not affiliated with IMAX Corporation or Dolby Laboratories. "IMAX" and "Dolby" are trademarks of their respective owners and are used here only to describe the kind of room being simulated.

---

## What it does

```
stereo music ──(optional) AI upmix──┐
                                    ▼
5.1 (e.g. lossless TrueHD from a Blu-ray) ──► virtual cinema ──► binaural stereo for headphones
```

- **Virtual room**: 30 × 24 × 18 m (≈13,000 m³). The seat is two-thirds of the way back.
- **Speakers**: screen speakers at ±37°. Surround arrays of 6 speakers per side, mounted 15° above ear height.
- **Bass**: 80 Hz bass management to a subwoofer on the floor below the screen.
- **Calibration**: cinema-style (surrounds −3 dB, LFE +10 dB in-band gain).
- **Room model**: image sources (3rd order) plus a statistical late tail, calibrated to RT60 = 0.5 s. It includes air absorption and horn directivity.
- **HRTF**: a symmetric Neumann KU 100 dummy head by default, with direct-sound-referenced timbre correction. Narrow pinna resonances are removed while notches are kept.
- **Output**: no limiter (attacks are never rounded). The level is set so the peak stays at or below −1 dBFS.
- **Speaker visualizer**: renders a video of the virtual room showing which speaker is playing and how loud (ripples).

## Listen (headphones)

**Video** (speaker ripples in the virtual theater + the cinebinaural audio): [[`samples/inmu_king_cinebinaural_ripple.mp4`](samples/inmu_king_cinebinaural_ripple.mp4)](https://github.com/user-attachments/assets/d5100eb5-73fe-4dd1-ab23-f78c568b451b
)

A 60 s excerpt (2:00–3:00) of a freely distributed track, loudness-matched (−16 LUFS) and not limited:

| | file |
|---|---|
| original stereo | [`samples/inmu_king_original.flac`](samples/inmu_king_original.flac) |
| **cinebinaural** (upmix `--center repan --ambience ls --surround-decorr --surround-delay-ms 10`, then `--preset standard4`) | [`samples/inmu_king_cinebinaural.flac`](samples/inmu_king_cinebinaural.flac) |

Track: "INMU KING" by やじゅまん (album *INMU KING / YAJU-MC*), distributed by the creator for free use. Source: https://www.youtube.com/watch?v=MFwtpM21wWc

## Quick start

```bash
pip install -r requirements.txt

# 5.1 wav (FL FR FC LFE SL SR) -> binaural
python -m cinebinaural.run process movie_5p1.wav out.wav --preset standard4 --no-limit

# stereo music -> 5.1 -> binaural
python -m cinebinaural.run upmix song.wav song_5p1.wav --center repan --ambience ls --surround-decorr --surround-delay-ms 10
python -m cinebinaural.run process song_5p1.wav song_binaural.wav --preset standard4 --no-limit

# long files (a full movie) without chunking. The output is bit-identical to in-memory processing.
python -m cinebinaural.run process movie_5p1.w64 movie.wav --preset standard4 --no-limit --lowmem --work-dir /big/disk/work

# speaker visualization video (ripples)
python -m cinebinaural.run video movie_5p1.wav out.wav speakers.mp4 --preset standard4 --view 35 90 --lang en
```

The input must be 48 kHz. Resample 44.1 kHz material first (e.g. `scipy.signal.resample_poly(x, 160, 147)`).

## HRTF data (download separately)

Put the SOFA files in the working directory. They are not redistributed here.

| name (`--hrtf`) | file | source |
|---|---|---|
| `ku100` (default in standard4) | `KU100_HRIR_FULL2DEG.sofa` | Neumann KU 100, TH Köln (Bernschütz 2013), [Zenodo 3928297](https://zenodo.org/records/3928297), CC BY 3.0 |
| `riec` | `RIEC_hrir_subject_005.sofa` | RIEC HRTF database, Tohoku University |
| `kemar` | `D2_HRIR_SOFA/D2_48K_24bit_256tap_FIR_SOFA.sofa` | SADIE II (KEMAR), University of York |
| `irc1040` | `IRC_1040_R_44100.sofa` | IRCAM LISTEN, [sofacoustics.org](https://sofacoustics.org/data/database/listen%20(hrtf)/). Resampled to 48 kHz on load; use `--hrtf-az-interp` |

Please follow each provider's license terms.

## Presets

| preset | adds on top of the previous one |
|---|---|
| `standard` | room model, array surrounds at +15°, bass management, low-frequency boundary gain, cinema calibration |
| `standard2` | caps early-reflection energy at the statistical prediction; notches room-mode peaks on the subwoofer |
| `standard3` | timbre correction referenced to the **direct sound** (Toole 2015) instead of the steady state |
| **`standard4`** (recommended) | KU 100 HRTF, no high-frequency shelf, removal of narrow pinna peaks from 5 to 16 kHz |

### Why standard4

Four problems were found by measurement, each with a fix:

- **Inconsistent tonal references**
  - standard3 combined Toole's direct-sound reference with an X-curve-like HF shelf, which darkened the sound twice. It measured +3.4 dB at 125 Hz and −3.9 dB at 4 kHz relative to the intended cinema spectrum.
  - Fix: remove the shelf, following Toole 2015.
- **Narrow pinna peaks**
  - A non-individual HRTF leaves narrow peaks of +3 to +6 dB between 6 and 16 kHz. You hear them as harshness, not as direction.
  - Fix: only the part above the 1/3-octave envelope is cut. Notches are kept. Elevation is carried mainly by N1, N2 and P1 (Iida et al. 2007).
- **Asymmetric ITDs**
  - The measured human HRTF had arrival times that differed by about 80 µs between mirrored directions.
  - When the same sound comes from L and R, the resulting comb-filter notches then fall at different frequencies in each ear. The two ears differed by ±6 to 8 dB between 1 and 4 kHz.
  - Fix: a symmetric dummy head (KU 100). Alternatively, mirror one ear with `--hrtf-symmetric left|right`.

## Upmixer (stereo → 5.1)

- **Separation**: Demucs `htdemucs_ft` (MPS / CUDA / CPU).
- **Center**: every stem, not just vocals, is re-panned over L-C-R (`--center repan`, after Avendano & Jot 2004).
  - This moves center-panned kick, bass and snare into the real center speaker, so they no longer play as a phantom center.
  - On test songs, the phantom-center share of L/R at 120–500 Hz dropped from 59–69% to 33–41%.
- **Ambience**: least-squares direct/ambient estimation (`--ambience ls`, after Faller 2006).
  - Front-to-surround coherence dropped from 0.9 to about 0.65.
- **Surround decorrelation and delay**: `--surround-decorr --surround-delay-ms 10`.
  - The decorrelator is flat in magnitude (±0.2 dB).
  - The 10 ms delay keeps the frontal image through the precedence effect (Haas 1951; Litovsky et al. 1999).
- **Downmix compatibility**: without the decorrelation and delay option, the ITU downmix of the result reconstructs the original stereo to −157 dB.
- `--stems-cache x.npz` reuses the separation, so option variants can be compared on identical stems.

## Reproducibility

- `--lowmem` renders a full-length film without chunking. It calls the same functions on the same full-length arrays and spills intermediates to disk (memory-mapped).
  - Its output is verified to be md5-identical to in-memory processing.
- Demucs on MPS is not bit-reproducible run to run (differences below −46 dB). On CPU with `shifts=0` it is.

## References (main ones)

- F. E. Toole, "The Measurement and Calibration of Sound Reproducing Systems," *JAES* 63(7/8), 2015
- K. Iida et al., "Median plane localization using a parametric model of the HRTF based on spectral cues," *Applied Acoustics* 68, 2007
- A. Kulkarni et al., "Sensitivity of human subjects to head-related transfer-function phase spectra," *JASA* 105(5), 1999
- C. Avendano, J.-M. Jot, "A Frequency-Domain Approach to Multichannel Upmix," *JAES* 52(7/8), 2004
- C. Faller, "Multiple-Loudspeaker Playback of Stereo Signals," *JAES* 54(11), 2006
- J. He, E.-L. Tan, W.-S. Gan, "Linear Estimation Based Primary-Ambient Extraction for Stereo Audio Signals," *IEEE/ACM TASLP* 22(2), 2014
- H. Haas, 1951 / R. Litovsky et al., "The precedence effect," *JASA* 106(4), 1999
- B. Bernschütz, "A Spherical Far Field HRIR/HRTF Compilation of the Neumann KU 100," DAGA 2013
- SMPTE ST 202 (B-chain / X-curve), SMPTE RP 200, ITU-R BS.775, ISO 9613-1, Dolby cinema technical guidelines, public IMAX patents (theater geometry, PPS loudspeakers)

The full design rationale, with every listening test, is in [`THEORY_FOUNDATION.md`](THEORY_FOUNDATION.md) (Japanese). These notes use the prototype's former package name, `imaxv2`.

## License

Code: MIT (see [LICENSE](LICENSE)). HRTF datasets are not included; each is subject to its provider's license (the KU 100 set is CC BY 3.0: TH Köln, B. Bernschütz).

---

## 日本語の概要

IMAX のような大型シアターの音響を、ヘッドホンで再現するツールです。映画の5.1(BD の TrueHD など)と、AIでアップミックスしたステレオ曲の両方に対応しています。

- 部屋の響き、スピーカー配置、劇場の校正、HRTF の処理はすべて論文と規格を根拠に決めました。
- 変更のたびに、音量(LUFS)をそろえた聴き比べで判断しています。
- おすすめ設定は `--preset standard4 --no-limit` です。リミッターを使わないので、音の立ち上がりが丸まりません。
