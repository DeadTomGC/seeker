"""Bench test: Pico loopback of a control frame over the RPi UART link."""
from machine import Pin, UART
import time
from uart_wrapper import *

uart = UART(1, baudrate=115200, tx=Pin(4), rx=Pin(5), timeout=0)
rx = Receiver(uart)

send_msg(uart, MSG_CONTROL, pack_control(CTRL_FLAG_SEEK_ACTIVE, TRACKING_LOCKED, 0, 25, -25))
res = rx.poll()
print(res)
if not res is None:
    if res[0] == MSG_TELEMETRY:
        print(unpack_telemetry(res[1]))
    elif res[0] == MSG_CONTROL:
        print(unpack_control(res[1]))
time.sleep(20)
