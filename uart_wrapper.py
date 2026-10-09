import struct

START = 0xAA
MAX_LEN = 128

STATUS_FMT = "<Bfff"  # little-endian: 1 unsigned byte + 3 x 32-bit floats = 13 bytes

def pack_status(status, a, b, c):
    return struct.pack(STATUS_FMT, status, a, b, c)

def unpack_status(payload):
    """Returns (status, a, b, c), or None if payload is the wrong size."""
    if len(payload) != struct.calcsize(STATUS_FMT):
        return None
    return struct.unpack(STATUS_FMT, payload)

def crc16(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE."""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc

def send_msg(uart, msg_type, payload=b""):
    """Frame and send a message. Returns False if payload too long."""
    if len(payload) > MAX_LEN:
        return False
    body = bytes([msg_type, len(payload)]) + payload
    crc = crc16(body)
    uart.write(bytes([START]) + body + bytes([crc >> 8, crc & 0xFF]))
    return True

class Receiver:
    """Call poll() often; returns (type, payload) when a valid packet completes."""
    def __init__(self, uart):
        self.uart = uart
        self.buf = bytearray()

    def poll(self):
        data = self.uart.read(64)  # non-blocking; may return None/empty
        if data:
            self.buf.extend(data)
        return self._parse()

    def _parse(self):
        while True:
            # Resync: discard everything before the next start byte
            i = self.buf.find(bytes([START]))
            if i < 0:
                self.buf = bytearray()
                return None
            self.buf = self.buf[i:]

            if len(self.buf) < 3:
                return None            # need START, TYPE, LEN
            length = self.buf[2]
            if length > MAX_LEN:
                self.buf = self.buf[1:]  # bogus length: skip this START, rescan
                continue

            total = 3 + length + 2
            if len(self.buf) < total:
                return None            # wait for more bytes

            body = bytes(self.buf[1:3 + length])
            rx_crc = (self.buf[3 + length] << 8) | self.buf[4 + length]
            if crc16(body) == rx_crc:
                msg = (self.buf[1], bytes(self.buf[3:3 + length]))
                self.buf = self.buf[total:]
                return msg
            self.buf = self.buf[1:]    # bad CRC: skip this START, rescan