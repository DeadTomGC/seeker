import machine
from machine import Pin, UART
import time
from bno055 import *
from control import *
from uart_wrapper import *

status_code = 0

uart = UART(1, baudrate=115200, tx=Pin(4), rx=Pin(5),timeout = 0)

rx = Receiver(uart)

# Pyboard hardware I2C
i2c = machine.I2C(1)
imu = BNO055(i2c,address=0x29)
calibrated = False
count = 0
pitch_command = 0
yaw_command = 0
cmd_rx_count = 0
error_count = 0
fail_safe = False
while True:
    time.sleep(0.02)
    try:
        gyro_data = imu.gyro()
        error_count = 0
    except:
        error_count += 1
        if error_count > 20:
            #IMU disconnected.
            #disable motor
            fail_safe = True
    pyr_output = update_pid(gyro_data,pitch_command,yaw_command)
    cmd_rx_count += 1
    if cmd_rx_count > 10: # check for stale command data, we expect it at ~30Hz
        pitch_command = 0
        yaw_command = 0
    servo_outputs = mix_controls(pyr_output[2],pyr_output[0],pyr_output[1])
    control_servos(servo_outputs)
    send_msg(uart,0x03,pack_status(0,gyro_data[2],gyro_data[0],gyro_data[1]))
    res = rx.poll()
    if not res is None:
        if res[0] == 0x03:
            data = unpack_status(res[1])
            if data[0] == 1:
                #enable primary motor
                pass
            if data[0] == 0:
                #disable primary motor
                pass
            pitch_command = data[2]
            yaw_command = data[3]
            cmd_rx_count = 0
        elif res[0] == 0x00:
            msg = res[1] #hello message?
    
    count += 1
    if count %20 == 0:
        count = 0
        if not calibrated:
            calibrated = imu.calibrated()
            print('Calibration required: sys {} gyro {} accel {} mag {}'.format(*imu.cal_status()))
        print('Temperature {}°C'.format(imu.temperature()))
        print('Controls  r {:5.0f}     p {:5.0f}     y {:5.0f}'.format(pyr_output[2],pyr_output[0],pyr_output[1]))
        print('Mag       x {:5.0f}    y {:5.0f}     z {:5.0f}'.format(*imu.mag()))
        print('Gyro      x {:5.0f}    y {:5.0f}     z {:5.0f}'.format(*imu.gyro()))
        print('Accel     x {:5.1f}    y {:5.1f}     z {:5.1f}'.format(*imu.accel()))
        print('Lin acc.  x {:5.1f}    y {:5.1f}     z {:5.1f}'.format(*imu.lin_acc()))
        print('Gravity   x {:5.1f}    y {:5.1f}     z {:5.1f}'.format(*imu.gravity()))
        print('Heading     {:4.0f} roll {:4.0f} pitch {:4.0f}'.format(*imu.euler()))