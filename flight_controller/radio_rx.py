"""CRSF (TBS / ExpressLRS) RC link for the flight controller Pico.

Thin adapter around the crsf package (flight_controller/crsf): pumps
frames from the ELRS receiver, exposes the latest RC channels and link
state, and queues telemetry for the TX.

Usage: call process_frames() from the control loop as often as possible.
"""

from machine import Pin, UART

from crsf import BAUD_RATE, CRSF, ticks_diff, ticks_ms

# Which UART the ELRS receiver hangs off: UART0 = GP0 (TX out to RX),
# GP1 (RX in from RX).  ELRS receivers output CRSF at 420000 baud.
DEFAULT_UART = 0
DEFAULT_TX_PIN = 0
DEFAULT_RX_PIN = 1

CRSF_CENTER = 992        # channel value at stick centre
CRSF_HALF_RANGE = 820    # channel value at 100% stick deflection
CRSF_TIMEOUT_MS = 500    # no RC frames for this long -> failsafe


class RadioRx:
    """RC channels + failsafe state from the ELRS RX, telemetry back to it."""

    def __init__(self, uart):
        self._crsf = CRSF(uart)
        self.channels = [CRSF_CENTER] * 16
        self._last_frame_ms = None
        self._crsf.set_on_rc_channels(self._on_channels)

    def _on_channels(self, channels):
        self.channels = list(channels)
        self._last_frame_ms = ticks_ms()

    @property
    def failsafe(self):
        """True while the RC link is down or degraded."""
        if self._last_frame_ms is None or \
           ticks_diff(ticks_ms(), self._last_frame_ms) >= CRSF_TIMEOUT_MS:
            return True
        return self._crsf.failsafe

    def process_frames(self):
        """Pump the UART: decode RC frames, send queued telemetry."""
        self._crsf.process_frames()

    def channel_percent(self, ch):
        """Channel value in -100..100 around centre."""
        pct = (self.channels[ch] - CRSF_CENTER) * 100.0 / CRSF_HALF_RANGE
        return max(-100.0, min(100.0, pct))

    # ---- telemetry -------------------------------------------------------

    def send_battery(self, voltage, current, capacity_mah, remaining_pct):
        """BATTERY_SENSOR telemetry.  Voltage in V, current in A."""
        self._crsf.telem_set_battery_data(
            int(voltage * 10), int(current * 10),
            int(capacity_mah), max(0, min(101, int(remaining_pct))))

    def send_attitude(self, pitch_deg, roll_deg, yaw_deg):
        """ATTITUDE telemetry.  Angles in degrees."""
        self._crsf.telem_set_attitude_data(pitch_deg, roll_deg, yaw_deg)

    def send_flight_mode(self, text):
        """FLIGHT_MODE telemetry: short status string."""
        self._crsf.telem_set_flight_mode(text)


def create_csrf():
    """Build the RC link on the default UART pins."""
    uart = UART(DEFAULT_UART, baudrate=BAUD_RATE,
                tx=Pin(DEFAULT_TX_PIN), rx=Pin(DEFAULT_RX_PIN), timeout=0)
    return RadioRx(uart)
