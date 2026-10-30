"""RPi-side UART smoke test: send an armed control frame, listen for telemetry."""
from flight_controller.uart_wrapper import *
import serial

uart = serial.Serial("/dev/serial0", 115200, timeout=0)  # timeout=0 -> non-blocking
rx = Receiver(uart)

send_msg(uart, MSG_CONTROL, pack_control(0, TRACKING_NONE, 0, 0, 0))
while True:
    res = rx.poll()
    if not res is None:
        if res[0] == MSG_TELEMETRY:
            print("telemetry:", unpack_telemetry(res[1]))
        elif res[0] == MSG_HELLO:
            print("hello:", unpack_hello(res[1]))
