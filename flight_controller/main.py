import machine
from machine import Pin, UART
import time
from bno055 import *
from control import *
from uart_wrapper import *
import radio_rx


# --- RC channel mapping (standard AIL/ELE/THR/RUD) -----------------------
CH_ROLL = 0
CH_PITCH = 1
CH_THROTTLE = 2
CH_YAW = 3
CH_MODE = 4     # 2-position switch: low = MANUAL, high = AUTOMATIC
CH_SUBMODE = 5  # 3-position switch (automatic mode): standby / search / seek
# Flip any of these if an axis moves the wrong way.
ROLL_SIGN = 1.0
PITCH_SIGN = 1.0
YAW_SIGN = 1.0

SWITCH_THRESHOLD = 25.0  # +-25% bands around switch detents
SUB_STANDBY = 0
SUB_SEARCH = 1
SUB_SEEK = 2

# --- blending ------------------------------------------------------------
RPI_TIMEOUT_MS = 500    # no RPi frames for this long -> seeker commands stale
SEEK_CMD_SCALE = 1.0   # gain on RPi seek offsets added to pilot sticks
CRSF_TELEM_PERIOD = 10 # send each telemetry type every N loop iterations

crsf = radio_rx.create_csrf()
uart = UART(1, baudrate=115200, tx=Pin(4), rx=Pin(5),timeout = 0)

rx = Receiver(uart)

# Pyboard hardware I2C
i2c = machine.I2C(1)
imu = BNO055(i2c,address=0x29)

# --- loop state ----------------------------------------------------------
count = 0
calibrated = False
imu_fail = False       # BNO055 not responding (cause bit for telemetry)
error_count = 0
gyro_data = (0.0, 0.0, 0.0)
euler = (0.0, 0.0, 0.0)
tracking_command = 0   # last reported RPi tracker state (TRACKING_*)
pitch_command = 0
yaw_command = 0
roll_command = 0
rpi_cmd_ms = None
rpi_flags = 0
armed = False
seek_active = False
pi_standby = False
hello_seq = 0
# Dummy battery values for CRSF telemetry (Pico has no fuel gauge)
BATTERY_NOMINAL_V = 12.4
BATTERY_CURRENT_A = 2.0
BATTERY_CAPACITY_MAH = 2200
BATTERY_REMAINING_PCT = 75


# --- helpers ---------------------------------------------------------------
# The main loop below reads top to bottom: read inputs (RC link, IMU, RPi
# messages, switches, pilot sticks), decide (status, mode, blending), act
# (servo/motor output), report (telemetry, debug).

def read_imu():
    """Sample the IMU.  Repeated read failures raise the failsafe."""
    global gyro_data, euler, error_count, imu_fail
    try:
        gyro_data = imu.gyro()
        error_count = 0
    except:
        error_count += 1
        if error_count > 20:
            imu_fail = True  # IMU disconnected: failsafe disables motor
    try:
        euler = imu.euler()  # (heading, roll, pitch) degrees
    except:
        euler = (0.0, 0.0, 0.0)


def rpi_link_state(now_ms):
    """Check RPi link freshness and zero its commands while stale.

    The tracking status and steering commands arrive redundantly in
    every control frame, so a missed message is healed by the next one;
    staleness only matters after the whole link is gone.  Returns True
    while frames are fresh."""
    global pitch_command, yaw_command, roll_command
    global rpi_flags, tracking_command
    rpi_link_ok = (rpi_cmd_ms is not None and
                   time.ticks_diff(now_ms, rpi_cmd_ms) <= RPI_TIMEOUT_MS)
    if not rpi_link_ok:  # stale/missing RPi commands, hands off the seeker
        pitch_command = 0
        yaw_command = 0
        roll_command = 0
        rpi_flags = 0
        tracking_command = 0
    return rpi_link_ok


def read_switches():
    """Mode and submode switch positions from the RC receiver."""
    mode_manual = crsf.channel_percent(CH_MODE) <= SWITCH_THRESHOLD
    sub_pct = crsf.channel_percent(CH_SUBMODE)
    if sub_pct > SWITCH_THRESHOLD:
        submode = SUB_SEEK
    elif sub_pct >= -SWITCH_THRESHOLD:
        submode = SUB_SEARCH
    else:
        submode = SUB_STANDBY
    return mode_manual, submode


def update_mode(mode_manual, submode, rpi_tracking, failsafe):
    """Mode state machine: sets pi_standby / seek_active / armed.

    MANUAL (mode switch low): pilot sticks only, RPi input ignored
    entirely (including a bad RPi link), motor armed whenever failsafe
    is off and connections (IMU, RC link) are good, and the RPi is told
    to stand by so it keeps resetting its tracks.
    AUTOMATIC (mode switch high), controlled by the 3-position switch:
      standby -> RPi stands by (resets tracks), motor disarmed
      search  -> RPi may track and steer, motor stays disarmed
      seek    -> engaged only while a track exists: RPi steers with
                 motor armed and the RPi 'go' function is activated
    """
    global pi_standby, seek_active, armed
    global pitch_command, yaw_command, roll_command, tracking_command
    if mode_manual:
        pi_standby = True
        seek_active = False
        armed = not failsafe
        pitch_command = 0
        yaw_command = 0
        roll_command = 0
        tracking_command = 0
    elif submode == SUB_STANDBY:
        pi_standby = True
        seek_active = False
        armed = False
    elif submode == SUB_SEARCH:
        pi_standby = False
        seek_active = False
        armed = False
    else:  # SUB_SEEK: only engaged while the RPi reports a live track
        pi_standby = False
        seek_active = rpi_tracking
        armed = seek_active and not failsafe


def read_pilot(rc_failsafe):
    """Pilot stick inputs in percent; RC link lost -> surfaces neutral."""
    if rc_failsafe:
        return 0.0, 0.0, 0.0, 0.0
    return (ROLL_SIGN * crsf.channel_percent(CH_ROLL),
            PITCH_SIGN * crsf.channel_percent(CH_PITCH),
            YAW_SIGN * crsf.channel_percent(CH_YAW),
            (crsf.channel_percent(CH_THROTTLE) + 100.0) / 2.0)


def blend_commands(pilot_roll, pilot_pitch, pilot_yaw, blend_ok):
    """Blend pilot sticks with the RPi's steering offsets, clamped.

    The seeker only contributes in automatic mode with the RPi link
    healthy and not in standby.  Steering is wanted whenever the RPi is
    tracking, including SEARCH (surfaces only, motor stays disarmed);
    SEEK additionally arms the motor."""
    global pitch_command, yaw_command, roll_command
    if not blend_ok:
        pitch_command = 0
        yaw_command = 0
        roll_command = 0
    roll_cmd = max(-100.0, min(100.0, pilot_roll + roll_command * SEEK_CMD_SCALE))
    pitch_cmd = max(-100.0, min(100.0, pilot_pitch + pitch_command * SEEK_CMD_SCALE))
    yaw_cmd = max(-100.0, min(100.0, pilot_yaw + yaw_command * SEEK_CMD_SCALE))
    return roll_cmd, pitch_cmd, yaw_cmd


def drive_outputs(roll_cmd, pitch_cmd, yaw_cmd, throttle):
    """Run the rate PID and drive the control surface servos and ESC."""
    pyr_output = update_pid(gyro_data, pitch_cmd, yaw_cmd, roll_cmd)
    servo_outputs = mix_controls(pyr_output[2], pyr_output[0], pyr_output[1])
    control_servos(servo_outputs)
    # While disarmed the ESC is held at a 1000 us idle pulse.
    control_throttle(throttle, armed)
    return pyr_output


def status_bits(rc_failsafe):
    """Status byte reported to the RPi."""
    status = 0
    if imu_fail:
        status |= STATUS_IMU_FAIL
    if rc_failsafe:
        status |= STATUS_RC_FAILSAFE
    if armed:
        status |= STATUS_MOTOR_ARMED
    if seek_active:
        status |= STATUS_SEEKING
        status |= STATUS_PI_GO
    if pi_standby:
        status |= STATUS_PI_STANDBY
    return status


def flight_mode_string(mode_manual, submode, rpi_tracking):
    """Flight mode string reported to the TX."""
    if mode_manual:
        return "MANUAL"
    if submode == SUB_STANDBY:
        return "STANDBY"
    if seek_active:
        return "SEEK"
    if rpi_tracking:
        return "TRACKING"
    return "SEARCH"


def send_telemetry(mode_str, rc_failsafe):
    """Telemetry to the RPi over the serial link and to the TX over CRSF."""
    send_msg(uart, MSG_TELEMETRY,
             pack_telemetry(status_bits(rc_failsafe),
                            euler[1], euler[2], euler[0],
                            gyro_data[0], gyro_data[1], gyro_data[2],
                            BATTERY_NOMINAL_V))
    if count % CRSF_TELEM_PERIOD == 0:
        crsf.send_attitude(euler[2], euler[1], euler[0])
        crsf.send_battery(BATTERY_NOMINAL_V, BATTERY_CURRENT_A,
                          BATTERY_CAPACITY_MAH, BATTERY_REMAINING_PCT)
    if count % (2 * CRSF_TELEM_PERIOD) == 0:
        crsf.send_flight_mode(mode_str)


def process_rpi_messages(now_ms):
    """Drain every pending message from the RPi link.

    Control frames update the steering commands and HELLO is answered
    to keep the handshake alive; anything else on the wire is ignored.
    Draining the whole buffer keeps a burst of frames from staling the
    link, and the newest control frame simply wins."""
    global tracking_command, roll_command, pitch_command, yaw_command
    global rpi_flags, rpi_cmd_ms, hello_seq
    while True:
        res = rx.poll()
        if res is None:
            return
        if res[0] == MSG_CONTROL:
            data = unpack_control(res[1])
            if data is None:
                continue  # malformed frame: drop it, keep draining
            rpi_flags = data[0]
            tracking_command = data[1]
            roll_command = data[2]
            pitch_command = data[3]
            yaw_command = data[4]
            rpi_cmd_ms = now_ms
        elif res[0] == MSG_HELLO:
            hello_seq += 1
            send_msg(uart, MSG_HELLO, pack_hello(hello_seq))


def debug_print(mode_str, rc_failsafe, channels, pyr_output):
    """Periodic console dump for bench debugging."""
    global calibrated
    print(channels)
    print('Mode      {}{}{} armed {}'.format(
        mode_str,
        ' FAILSAFE' if imu_fail else '',
        ' RC-FAILSAFE' if rc_failsafe else '',
        armed))
    print('Controls  r {:5.0f}     p {:5.0f}     y {:5.0f}'.format(
        pyr_output[2], pyr_output[0], pyr_output[1]))
    try:
        if not calibrated:
            calibrated = imu.calibrated()
            print('Calibration required: sys {} gyro {} accel {} mag {}'.format(*imu.cal_status()))
        print('Temperature {}°C'.format(imu.temperature()))
        print('Mag       x {:5.0f}    y {:5.0f}     z {:5.0f}'.format(*imu.mag()))
        print('Gyro      x {:5.0f}    y {:5.0f}     z {:5.0f}'.format(*imu.gyro()))
        print('Accel     x {:5.1f}    y {:5.1f}     z {:5.1f}'.format(*imu.accel()))
        print('Lin acc.  x {:5.1f}    y {:5.1f}     z {:5.1f}'.format(*imu.lin_acc()))
        print('Gravity   x {:5.1f}    y {:5.1f}     z {:5.1f}'.format(*imu.gravity()))
        print('Heading     {:4.0f} roll {:4.0f} pitch {:4.0f}'.format(*imu.euler()))
    except:
        print('IMU read failed')


# --- main loop -------------------------------------------------------------
# Reads first (RC link, IMU, RPi messages, switches, pilot sticks), then
# status/mode decisions and blending, then actions (servos, ESC), then
# telemetry and debug output.

while True:
    # --- read inputs -------------------------------------------------------
    crsf.process_frames()
    channels = crsf.channels
    now_ms = time.ticks_ms()
    read_imu()
    process_rpi_messages(now_ms)
    rc_failsafe = crsf.failsafe
    mode_manual, submode = read_switches()
    pilot_roll, pilot_pitch, pilot_yaw, throttle = read_pilot(rc_failsafe)

    # --- check status / decide ---------------------------------------------
    # One failsafe flag for every failure type: any of them fails safe.
    # The cause is still reported bit-by-bit in telemetry / debug output.
    failsafe = imu_fail or rc_failsafe
    rpi_link_ok = rpi_link_state(now_ms)
    rpi_tracking = rpi_link_ok and tracking_command == TRACKING_LOCKED
    update_mode(mode_manual, submode, rpi_tracking, failsafe)
    blend_ok = (not mode_manual) and rpi_link_ok and not pi_standby
    roll_cmd, pitch_cmd, yaw_cmd = blend_commands(
        pilot_roll, pilot_pitch, pilot_yaw, blend_ok)

    # --- act ---------------------------------------------------------------
    pyr_output = drive_outputs(roll_cmd, pitch_cmd, yaw_cmd, throttle)

    # --- report ------------------------------------------------------------
    mode_str = flight_mode_string(mode_manual, submode, rpi_tracking)
    send_telemetry(mode_str, rc_failsafe)
    count += 1
    if count % (5*CRSF_TELEM_PERIOD) == 0:
        debug_print(mode_str, rc_failsafe, channels, pyr_output)
