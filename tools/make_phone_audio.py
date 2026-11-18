"""
Generate phone-optimized versions of the FSK command WAVs.

Phone speakers are weak around the low FSK tone (781 Hz) compared to the high
tone (1562 Hz). Because FSK only ever plays ONE tone at a time, we can process
each tone independently:

  * Low tone (781 Hz):  soft-clipped (tanh) toward a square wave. A square wave's
    fundamental is 4/pi (~+2.1 dB) above a full-scale sine, and its harmonics are
    ODD only (2343, 3905 Hz, ...), so nothing lands on the 1562 Hz tone.
  * High tone (1562 Hz): kept a clean sine, normalized to near full scale.

Segments are split at zero crossings (where the FSK tone switches), so the gain
change never causes a click.

Usage:  python tools/make_phone_audio.py
Input:  audio_cmds/computer/*.wav      Output: audio_cmds/phone/*.wav
"""

from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "audio_cmds" / "computer"
OUT_DIR = ROOT / "audio_cmds" / "phone"

LOW_HZ = 781.25           # 50 kHz / 64
HIGH_HZ = 1562.5          # 50 kHz / 32
HIGH_PEAK = 0.98          # high tone peak, fraction of full scale
LOW_DRIVE = 4.0           # tanh drive for low tone; higher = more square, louder
LOW_PEAK = 0.98           # low tone peak after clipping
SILENCE = 1e-4            # samples below this (normalized) are treated as silence


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path.name}: expected 16-bit mono PCM")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return sr, data.astype(np.float64) / 32768.0


def write_wav(path: Path, sr: int, x: np.ndarray) -> None:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def low_tone_mask(x: np.ndarray, sr: int) -> np.ndarray:
    """True for samples inside a half-cycle of the low tone (split at zero crossings)."""
    mask = np.zeros(len(x), dtype=bool)
    active = np.abs(x) > SILENCE
    sign = np.signbit(x)
    edges = np.nonzero(np.diff(sign))[0] + 1
    bounds = np.concatenate(([0], edges, [len(x)]))
    # Half-period threshold halfway between the two tones' half-periods
    threshold = 0.5 * (sr / (2 * LOW_HZ) + sr / (2 * HIGH_HZ))
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start > threshold and active[start:end].any():
            mask[start:end] = True
    return mask


def process(x: np.ndarray, sr: int) -> np.ndarray:
    peak = np.max(np.abs(x))
    if peak == 0:
        raise ValueError("file is silent")
    unit = x / peak                                   # tones now peak at 1.0
    low = LOW_PEAK * np.tanh(LOW_DRIVE * unit) / np.tanh(LOW_DRIVE)
    high = HIGH_PEAK * unit
    return np.where(low_tone_mask(x, sr), low, high)


def tone_amplitude(x: np.ndarray, sr: int, hz: float, mask: np.ndarray) -> float:
    """Amplitude of the component at `hz` within the masked samples (via projection)."""
    seg = x[mask]
    t = np.nonzero(mask)[0] / sr
    c = 2 * np.mean(seg * np.cos(2 * np.pi * hz * t))
    s = 2 * np.mean(seg * np.sin(2 * np.pi * hz * t))
    return np.hypot(c, s)


def db(ratio: float) -> float:
    return 20 * np.log10(ratio)


def main() -> None:
    files = sorted(SRC_DIR.glob("*.wav"))
    if not files:
        sys.exit(f"No WAV files found in {SRC_DIR}")
    OUT_DIR.mkdir(exist_ok=True)

    for path in files:
        sr, x = read_wav(path)
        y = process(x, sr)
        write_wav(OUT_DIR / path.name, sr, y)

        mask = low_tone_mask(x, sr)
        hi_mask = (~mask) & (np.abs(x) > SILENCE)
        line = f"{path.name}:"
        if mask.any():
            line += f" 781 Hz {db(tone_amplitude(y, sr, LOW_HZ, mask) / tone_amplitude(x, sr, LOW_HZ, mask)):+.1f} dB"
        if hi_mask.any():
            line += f", 1562 Hz {db(tone_amplitude(y, sr, HIGH_HZ, hi_mask) / tone_amplitude(x, sr, HIGH_HZ, hi_mask)):+.1f} dB"
        print(line)


if __name__ == "__main__":
    main()
