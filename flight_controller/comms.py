from machine import Pin, UART
from uart_wrapper import *
uart = UART(1, baudrate=115200, tx=Pin(4), rx=Pin(5))

rx = Receiver(uart)

send_msg(uart,0x03,pack_status(0,1,1,1))
res = rx.poll()
print(res)
if not res is None:
    data = unpack_status(res[1])
    print(data)
time.sleep(20)