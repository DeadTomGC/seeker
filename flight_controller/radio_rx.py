"""CRSF (TBS / ExpressLRS) link for the flight controller Pico.

Frame format:  <ADDR> <LEN> <TYPE> <PAYLOAD> <CRC8>
  - ADDR:  0xC8 (flight controller / sync byte) in both directions on this link
  - LEN:   byte count of TYPE + PAYLOAD + CRC8
  - CRC8:  DVB-S2 polynomial 0xD5 over TYPE + PAYLOAD

Inbound (ELRS RX -> Pico):
  0x16 RC_CHANNELS_PACKED   16 channels, 11 bits each, LSB first

Outbound (Pico -> ELRS RX -> air -> TX):
  0x08 BATTERY_SENSOR       voltage/current/capacity/remaining
  0x1E ATTITUDE             pitch/roll/yaw, int16 radians * 10000
  0x1F FLIGHT_MODE          short text status string

Usage: call process_frames() from the control loop as often as possible.
"""

from machine import Pin, UART
import struct
import time

# --- CRSF constants ----------------------------------------------------
CRSF_SYNC_BYTE = 0xC8          # flight controller address / sync byte
CRSF_MAX_FRAME_LEN = 64        # max value of the LEN byte we accept for TX
CRSF_RX_MAX_LEN = 26           # inbound frames (RC channels only): LEN = 24

CRSF_FRAMETYPE_BATTERY_SENSOR = 0x08
CRSF_FRAMETYPE_RC_CHANNELS_PACKED = 0x16
CRSF_FRAMETYPE_ATTITUDE = 0x1E
CRSF_FRAMETYPE_FLIGHT_MODE = 0x1F

RC_CHANNELS_PAYLOAD_LEN = 22   # 16 channels * 11 bits
RC_CHANNELS_TYPE = 0x16

CRSF_CENTER = 992              # channel value at stick centre
CRSF_HALF_RANGE = 820          # channel value at 100% stick deflection
CRSF_TIMEOUT_MS = 500          # no RC frames for this long -> failsafe

# Which UART the ELRS receiver hangs off: UART0 = GP0 (TX out to RX),
# GP1 (RX in from RX).  ELRS receivers output CRSF at 420000 baud.
DEFAULT_UART = 0
DEFAULT_TX_PIN = 0
DEFAULT_RX_PIN = 1
DEFAULT_BAUD = 420000

_DEG_TO_CRAD = 10000.0 / 57.295779513  # degrees -> radians * 10000


def crc8_dvb_s2(data, crc=0x00):
    """CRC-8/DVB-S2 as required by CRSF."""
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0xD5) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class CRSF:
    """Parses RC channels off the ELRS RX and pushes telemetry back to it."""

    def __init__(self, uart_id=DEFAULT_UART, tx=DEFAULT_TX_PIN,
                 rx=DEFAULT_RX_PIN, baud=DEFAULT_BAUD, failsafe_ms=CRSF_TIMEOUT_MS):
        self.uart = UART(uart_id, baudrate=baud, tx=Pin(tx), rx=Pin(rx), timeout=0)
        self.buf = bytearray()
        self.channels = [CRSF_CENTER] * 16
        self.failsafe = True        # RC link state
        self._failsafe_ms = failsafe_ms
        self._last_frame_ms = None
        self.frame_count = 0
        self.bad_frames = 0

    # ---- inbound -------------------------------------------------------

    def process_frames(self):
        """Pump the UART, decode any complete CRSF frames, update fail-safe."""
        data = self.uart.read(256)   # non-blocking
        if data:
            self.buf.extend(data)
        self._parse()

        if self._last_frame_ms is not None and \
           time.ticks_diff(time.ticks_ms(), self._last_frame_ms) < self._failsafe_ms:
            self.failsafe = False
        else:
            self.failsafe = True

    def _parse(self):
        buf = self.buf
        while True:
            if len(buf) < 2:
                break
            # Frame must start on the flight-controller sync byte
            if buf[0] != CRSF_SYNC_BYTE:
                del buf[0]
                continue
            length = buf[1]
            if not (2 <= length <= CRSF_RX_MAX_LEN):
                del buf[0]
                continue
            total = length + 2
            if len(buf) < total:
                break

            ftype = buf[2]
            payload = bytes(buf[3:1 + length])
            if crc8_dvb_s2(buf[2:1 + length]) == buf[1 + length]:
                if ftype == CRSF_FRAMETYPE_RC_CHANNELS_PACKED and \
                   len(payload) == RC_CHANNELS_PAYLOAD_LEN:
                    self._unpack_channels(payload)
                    self._last_frame_ms = time.ticks_ms()
                    self.frame_count += 1
                del buf[:total]
            else:
                self.bad_frames += 1
                del buf[0]           # resync one byte
        self.buf = buf

    def _unpack_channels(self, payload):
        bits = int.from_bytes(payload, "little")
        for i in range(16):
            self.channels[i] = (bits >> (11 * i)) & 0x7FF

    def channel_percent(self, ch):
        """Channel value in -100..100 around centre."""
        pct = (self.channels[ch] - CRSF_CENTER) * 100.0 / CRSF_HALF_RANGE
        return max(-100.0, min(100.0, pct))

    # ---- outbound telemetry --------------------------------------------

    def _send(self, ftype, payload):
        body = bytes([CRSF_SYNC_BYTE, len(payload) + 2, ftype]) + payload
        self.uart.write(body + bytes([crc8_dvb_s2(body[2:])]))

    def send_battery(self, voltage, current, capacity_mah, remaining_pct):
        """BATTERY_SENSOR frame.  Voltage in V, current in A."""
        payload = struct.pack("<HH", int(voltage * 100) & 0xFFFF,
                              int(current * 100) & 0xFFFF)
        payload += struct.pack("<I", int(capacity_mah) & 0xFFFFFF)[:3]
        payload += bytes([max(0, min(101, int(remaining_pct)))])
        self._send(CRSF_FRAMETYPE_BATTERY_SENSOR, payload)

    def send_attitude(self, pitch_deg, roll_deg, yaw_deg):
        """ATTITUDE frame, int16 radians * 10000 per CRSF spec."""
        vals = []
        for angle in (pitch_deg, roll_deg, yaw_deg):
            v = int(angle * _DEG_TO_CRAD)
            v = max(-32768, min(32767, v))
            vals.append(v)
        self._send(CRSF_FRAMETYPE_ATTITUDE, struct.pack("<hhh", *vals))

    def send_flight_mode(self, text):
        """FLIGHT_MODE frame: a short status string, max 16 chars."""
        self._send(CRSF_FRAMETYPE_FLIGHT_MODE, text[:16].encode() + b"\x00")


def create_csrf(**kwargs):
    """Factory kept for backwards compatibility with existing imports."""
    return CRSF(**kwargs)
