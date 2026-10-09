from uart_wrapper import *

import serial
uart = serial.Serial("/dev/serial0", 115200, timeout=0)  # timeout=0 -> non-blocking
rx = Receiver(uart)

send_msg(uart,0x01,b"hello")