from machine import Pin, UART
from crsf import BAUD_RATE, CRSF, ticks_to_us


def on_channels(channels):
    #TODO: real code
    print("Channel 1 (µs)", ticks_to_us(channels[0]))


def on_link(stats):
    #TODO: real code
    #print("RSSI:", stats.rssi, "dBm", "LQ:", stats.link_quality, "%")
    pass


def on_failsafe(state):
    #TODO: real code
    if state:
        print("Failsafe engaged")

def create_csrf():
    
    crsf_uart = UART(0, baudrate=BAUD_RATE, tx=Pin(16), rx=Pin(17),timeout = 0)
    crsf = CRSF(crsf_uart)
    crsf.set_on_rc_channels(on_channels)
    crsf.set_on_link_statistics(on_link)
    crsf.set_on_failsafe(on_failsafe)
    return crsf