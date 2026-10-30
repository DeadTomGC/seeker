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
calibrated = False
count = 0
tracking_command = 0   # last reported RPi tracker state (TRACKING_*)
pitch_command = 0
yaw_command = 0
roll_command = 0
rpi_cmd_ms = None
rpi_flags = 0
error_count = 0
fail_safe = False
gyro_data = (0.0, 0.0, 0.0)
euler = (0.0, 0.0, 0.0)
armed = False
hello_seq = 0
# Dummy battery values for CRSF telemetry (Pico has no fuel gauge)
BATTERY_NOMINAL_V = 12.4
BATTERY_CURRENT_A = 2.0
BATTERY_CAPACITY_MAH = 2200
BATTERY_REMAINING_PCT = 75

while True:
    crsf.process_frames()
    channels = crsf.channels
    now_ms = time.ticks_ms()
    #time.sleep(0.02)
    try:
        gyro_data = imu.gyro()
        error_count = 0
    except:
        error_count += 1
        if error_count > 20:
            #IMU disconnected.
            #disable motor
            fail_safe = True
    try:
        euler = imu.euler()  # (heading, roll, pitch) degrees
    except:
        euler = (0.0, 0.0, 0.0)
    #TODO: yall need jesus, and functionization
    # --- RC link state ----------------------------------------------------
    failsafe = fail_safe or crsf.failsafe

    # RPi link freshness: the tracking status and steering commands arrive
    # redundantly in every control frame, so a missed message is healed by
    # the next one; staleness only matters after the whole link is gone.
    rpi_link_ok = (rpi_cmd_ms is not None and
                   time.ticks_diff(now_ms, rpi_cmd_ms) <= RPI_TIMEOUT_MS)
    if not rpi_link_ok: # stale/missing RPi commands, hands off the seeker
        pitch_command = 0
        yaw_command = 0
        roll_command = 0
        rpi_flags = 0
        tracking_command = 0
    rpi_tracking = rpi_link_ok and tracking_command == TRACKING_LOCKED

    # --- switch positions -------------------------------------------------
    mode_manual = crsf.channel_percent(CH_MODE) <= SWITCH_THRESHOLD
    sub_pct = crsf.channel_percent(CH_SUBMODE)
    if sub_pct > SWITCH_THRESHOLD:
        submode = SUB_SEEK
    elif sub_pct >= -SWITCH_THRESHOLD:
        submode = SUB_SEARCH
    else:
        submode = SUB_STANDBY

    # --- mode state machine ----------------------------------------------
    # MANUAL (mode switch low): pilot sticks only, RPi input ignored
    # entirely (including a bad RPi link), motor armed whenever failsafe
    # is off and connections (IMU, RC link) are good, and the RPi is told
    # to stand by so it keeps resetting its tracks.
    # AUTOMATIC (mode switch high), controlled by the 3-position switch:
    #   standby -> RPi stands by (resets tracks), motor disarmed
    #   search  -> RPi may track and steer, motor stays disarmed
    #   seek    -> engaged only while a track exists: RPi steers with
    #              motor armed and the RPi 'go' function is activated
    pi_standby = False
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
    else: # SUB_SEEK: only engaged while the RPi reports a live track
        pi_standby = False
        seek_active = rpi_tracking
        armed = seek_active and not failsafe
    #TODO: consider using a single failsafe flag for all types of failures
    # --- pilot inputs from the CRSF receiver ------------------------------
    if crsf.failsafe:
        # RC link lost: hands off the controls, surfaces neutral
        pilot_roll = 0.0
        pilot_pitch = 0.0
        pilot_yaw = 0.0
        throttle = 0.0
    else:
        pilot_roll = ROLL_SIGN * crsf.channel_percent(CH_ROLL)
        pilot_pitch = PITCH_SIGN * crsf.channel_percent(CH_PITCH)
        pilot_yaw = YAW_SIGN * crsf.channel_percent(CH_YAW)
        throttle = (crsf.channel_percent(CH_THROTTLE) + 100.0) / 2.0

    # Blend: pilot stick plus the seeker's steering offsets, clamped.  The
    # seeker only contributes in automatic mode with the RPi link healthy
    # and not in standby.
    #TODO: double check that we want inputs when not seeking. (this is probably still right)
    blend_ok = (not mode_manual) and rpi_link_ok and not pi_standby
    if not blend_ok:
        pitch_command = 0
        yaw_command = 0
        roll_command = 0
    roll_cmd = max(-100.0, min(100.0, pilot_roll + roll_command * SEEK_CMD_SCALE))
    pitch_cmd = max(-100.0, min(100.0, pilot_pitch + pitch_command * SEEK_CMD_SCALE))
    yaw_cmd = max(-100.0, min(100.0, pilot_yaw + yaw_command * SEEK_CMD_SCALE))

    pyr_output = update_pid(gyro_data,pitch_cmd,yaw_cmd,roll_cmd)
    servo_outputs = mix_controls(pyr_output[2],pyr_output[0],pyr_output[1])
    control_servos(servo_outputs)

    # --- motor output -----------------------------------------------------
    # While disarmed the ESC is held at a 1000 us idle pulse.
    control_throttle(throttle, armed)

    # --- status / telemetry ---------------------------------------------
    status = 0
    if fail_safe:
        status |= STATUS_IMU_FAIL
    if crsf.failsafe:
        status |= STATUS_RC_FAILSAFE
    if armed:
        status |= STATUS_MOTOR_ARMED
    if seek_active:
        status |= STATUS_SEEKING
        status |= STATUS_PI_GO
    if pi_standby:
        status |= STATUS_PI_STANDBY

    # Flight mode string reported to the TX
    if mode_manual:
        mode_str = "MANUAL"
    elif submode == SUB_STANDBY:
        mode_str = "STANDBY"
    elif seek_active:
        mode_str = "SEEK"
    elif rpi_tracking:
        mode_str = "TRACKING"
    else:
        mode_str = "SEARCH"

    # Telemetry to the RPi over the serial link
    send_msg(uart,MSG_TELEMETRY,pack_telemetry(status,euler[1],euler[2],euler[0],
                                               gyro_data[0],gyro_data[1],gyro_data[2],
                                               BATTERY_NOMINAL_V))

    # Telemetry to the TX over the CRSF link
    if count % CRSF_TELEM_PERIOD == 0:
        crsf.send_attitude(euler[2],euler[1],euler[0])
        crsf.send_battery(BATTERY_NOMINAL_V,BATTERY_CURRENT_A,BATTERY_CAPACITY_MAH,BATTERY_REMAINING_PCT)
    if count % (2*CRSF_TELEM_PERIOD) == 0:
        crsf.send_flight_mode(mode_str)

    res = rx.poll() #TODO: Should we ensure that all incoming messages are processed?
    if not res is None:
        if res[0] == MSG_CONTROL:
            data = unpack_control(res[1])
            if data is None:
                continue #TODO: this assumes nothing else happens in the loop after this point
            rpi_flags = data[0]
            tracking_command = data[1]
            roll_command = data[2]
            pitch_command = data[3]
            yaw_command = data[4]
            rpi_cmd_ms = time.ticks_ms()
        elif res[0] == MSG_HELLO:
            hello_seq += 1
            send_msg(uart,MSG_HELLO,pack_hello(hello_seq))

    count += 1
    if count % (5*CRSF_TELEM_PERIOD) == 0:
        print(channels)
        print('Mode      {}{}{} armed {}'.format(
            mode_str,
            ' FAILSAFE' if fail_safe else '',
            ' RC-FAILSAFE' if crsf.failsafe else '',
            armed))
        print('Controls  r {:5.0f}     p {:5.0f}     y {:5.0f}'.format(pyr_output[2],pyr_output[0],pyr_output[1]))
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
