"""UART link to the flight-controller Pico over /dev/serial0.

Sends steering control frames (MSG_CONTROL) at ~30 Hz and receives
telemetry frames (MSG_TELEMETRY) back, using the shared framing in
flight_controller/uart_wrapper.py.

Control authority note: the Pico is the primary controller.  Arming and
mode selection are decided by the Pico from the CRSF switches; this link
only carries steering offsets and the tracker's tracking state.  Every
control frame redundantly includes the tracking status so missed frames
self-heal on the Pico side.
"""

import threading
import time

import serial

from flight_controller.uart_wrapper import (
    Receiver, send_msg, pack_control, pack_hello, unpack_telemetry,
    MSG_CONTROL, MSG_HELLO, MSG_TELEMETRY,
    CTRL_FLAG_SEEK_ACTIVE,
    TRACKING_NONE, TRACKING_LOCKED,
    STATUS_IMU_FAIL, STATUS_RC_FAILSAFE, STATUS_MOTOR_ARMED, STATUS_SEEKING,
    STATUS_PI_STANDBY, STATUS_PI_GO)

PORT = "/dev/serial0"
BAUD = 115200
CONTROL_RATE_HZ = 30.0
HELLO_INTERVAL_S = 2.0


class PicoLink:
    def __init__(self, port=PORT, baud=BAUD, silent=True):
        self.silent = silent
        self.ser = serial.Serial(port, baud, timeout=0)
        self.rx = Receiver(self.ser)

        self._lock = threading.Lock()
        self._cmd = (0, 0, 0.0, 0.0, 0.0)       # (flags, tracking, roll, pitch, yaw)
        self.telemetry = None                   # latest telemetry tuple
        self.telemetry_time = None
        self.hello_count = 0

        self._stop = False
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._tx_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_command(self, flags, tracking, roll, pitch, yaw):
        """Queue a steering frame: flags (CTRL_FLAG_*), tracking (TRACKING_*),
        roll/pitch/yaw -100..100 rate commands."""
        with self._lock:
            self._cmd = (int(flags), int(tracking), float(roll), float(pitch), float(yaw))

    def get_telemetry(self):
        """Latest telemetry dict or None: status bits, euler, gyro, battery,
        and the Pico-commanded RPi state (pi_standby / pi_go)."""
        with self._lock:
            t = self.telemetry
            t_time = self.telemetry_time
        if t is None:
            return None
        status, roll, pitch, yaw, gx, gy, gz, battery_v = t
        return {
            "status": status,
            "imu_fail": bool(status & STATUS_IMU_FAIL),
            "rc_failsafe": bool(status & STATUS_RC_FAILSAFE),
            "motor_armed": bool(status & STATUS_MOTOR_ARMED),
            "seeking": bool(status & STATUS_SEEKING),
            "pi_standby": bool(status & STATUS_PI_STANDBY),
            "pi_go": bool(status & STATUS_PI_GO),
            "euler": (roll, pitch, yaw),
            "gyro": (gx, gy, gz),
            "battery_v": battery_v,
            "age_s": time.time() - t_time,
        }

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def _send_loop(self):
        interval = 1.0 / CONTROL_RATE_HZ
        last_hello = 0.0
        while not self._stop:
            loop_start = time.time()
            with self._lock:
                cmd = self._cmd
            try:
                send_msg(self.ser, MSG_CONTROL, pack_control(*cmd))
                if time.time() - last_hello > HELLO_INTERVAL_S:
                    send_msg(self.ser, MSG_HELLO, pack_hello(cmd[0]))
                    last_hello = time.time()
            except (OSError, ValueError):
                pass  # port vanished; keep trying
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _receive_loop(self):
        while not self._stop:
            try:
                res = self.rx.poll()
                if res is None:
                    time.sleep(0.002)
                    continue
                if res[0] == MSG_TELEMETRY:
                    data = unpack_telemetry(res[1])
                    if data is not None:
                        with self._lock:
                            self.telemetry = data
                            self.telemetry_time = time.time()
                elif res[0] == MSG_HELLO:
                    with self._lock:
                        self.hello_count += 1
                    if not self.silent:
                        print("[PicoLink] hello from pico")
            except Exception as e:
                if not self.silent:
                    print(f"[PicoLink] rx error: {e}")
                time.sleep(0.1)


def steer_to_location(search_location, frame_size, max_cmd=100.0,
                      deadband_px=10.0, gain=0.15, silent=True):
    """Proportional steering law: keep the tracked search location centred.

    search_location is the tracker's image-space (x, y).  Returns
    (seek_active, roll, pitch, yaw) rate commands in -100..100, with roll
    always 0 (the RPi does not steer roll; the pilot does).

    Signs are set for a conventional layout and may need flipping per airframe.
    """
    offset_x = search_location[0] - frame_size[0] / 2.0
    offset_y = search_location[1] - frame_size[1] / 2.0

    if abs(offset_x) < deadband_px and abs(offset_y) < deadband_px:
        return False, 0.0, 0.0, 0.0

    # Target right of centre -> yaw right (positive); below centre -> pitch.
    yaw = max(-max_cmd, min(max_cmd, offset_x * gain))
    pitch = max(-max_cmd, min(max_cmd, offset_y * gain))
    return True, 0.0, pitch, yaw
