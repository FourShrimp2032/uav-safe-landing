"""Benewake TF-Luna serial protocol (UART mode, the factory default).

The sensor streams 9-byte frames at 115200 baud:

    0x59 0x59 Dist_L Dist_H Amp_L Amp_H Temp_L Temp_H Checksum

Distance is in cm, Amp is the signal strength and Checksum is the low byte
of the sum of the first 8 bytes. With a weak return (dark ground, strong
sunlight, out of range) Amp drops below 100 and the distance can't be
trusted; with an overexposed one it reads 65535.

Kept free of ROS and pyserial so it can be unit-tested anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

HEADER = 0x59
FRAME_LEN = 9
MIN_AMP = 100
OVEREXPOSED_AMP = 65535
MIN_RANGE_M = 0.2
MAX_RANGE_M = 8.0     # datasheet maximum, indoors on a white target; less outdoors


@dataclass
class Reading:
    distance_m: float
    amp: int
    temp_c: float

    @property
    def valid(self) -> bool:
        return (MIN_AMP <= self.amp < OVEREXPOSED_AMP
                and MIN_RANGE_M <= self.distance_m <= MAX_RANGE_M)


def parse_frame(frame: bytes) -> Reading | None:
    """One 9-byte frame -> Reading, or None if it isn't a valid frame."""
    if len(frame) != FRAME_LEN or frame[0] != HEADER or frame[1] != HEADER:
        return None
    if sum(frame[:8]) & 0xFF != frame[8]:
        return None
    dist = frame[2] | frame[3] << 8
    amp = frame[4] | frame[5] << 8
    temp = frame[6] | frame[7] << 8
    return Reading(dist / 100.0, amp, temp / 8.0 - 256.0)


class FrameParser:
    """Turns a byte stream into readings. Survives garbage and split frames:
    it looks for the 0x59 0x59 header and drops frames with a bad checksum."""

    def __init__(self):
        self.buf = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes) -> list[Reading]:
        self.buf.extend(data)
        out = []
        while True:
            i = self.buf.find(bytes((HEADER, HEADER)))
            if i < 0:
                # keep a trailing 0x59, it may be the first half of a header
                del self.buf[:max(0, len(self.buf) - 1)]
                return out
            del self.buf[:i]
            if len(self.buf) < FRAME_LEN:
                return out
            r = parse_frame(bytes(self.buf[:FRAME_LEN]))
            if r is None:
                self.bad_frames += 1
                del self.buf[:1]          # not a real header, search again
                continue
            out.append(r)
            del self.buf[:FRAME_LEN]


def make_frame(distance_cm: int, amp: int = 1000, temp_raw: int = 2248) -> bytes:
    """Build a frame (tests and bench tools)."""
    body = bytes((HEADER, HEADER, distance_cm & 0xFF, distance_cm >> 8 & 0xFF,
                  amp & 0xFF, amp >> 8 & 0xFF, temp_raw & 0xFF, temp_raw >> 8 & 0xFF))
    return body + bytes((sum(body) & 0xFF,))
