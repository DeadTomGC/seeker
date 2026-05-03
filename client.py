import socket
import threading
import json
import time
import random  # used only in the demo main loop
import base64
import numpy as np
import os
import uuid
# --- Configuration ---
SERVER_HOST = '192.168.1.249'  # <-- Set to your server's static IP
SERVER_PORT = 65432
RECONNECT_DELAY = 3.0          # seconds between reconnect attempts

def _load_or_create_client_id(id_file="client_id.txt"):
    """Load persistent client ID from file, or generate and save a new one."""
    if os.path.exists(id_file):
        with open(id_file, 'r') as f:
            return f.read().strip()
    new_id = str(uuid.uuid4())
    with open(id_file, 'w') as f:
        f.write(new_id)
    return new_id

class StatusClient:
    def __init__(self, host, port, silent = True):
        self.silent = silent
        self.client_id = _load_or_create_client_id()
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self._lock = threading.Lock()
        self._recv_buffer = ""
        self.go_callback = None   # set this to a callable to handle "go" commands

        # Start connection thread
        self._conn_thread = threading.Thread(target=self._connection_loop, daemon=True)
        self._conn_thread.start()

        # Start receive thread
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()

    def set_go_callback(self, callback):
        """Register a function to be called when the server sends 'go'."""
        self.go_callback = callback

    def _connection_loop(self):
        """Continuously try to connect/reconnect to the server."""
        while True:
            if not self.connected:
                try:
                    if not self.silent:
                        print(f"[Client] Connecting to {self.host}:{self.port}...")
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(5)
                    s.connect((self.host, self.port))
                    s.settimeout(0.1)
                    with self._lock:
                        self.sock = s
                        self.connected = True
                        self._send_handshake()
                    if not self.silent:
                        print(f"[Client] Connected to {self.host}:{self.port}")
                except Exception as e:
                    if not self.silent:
                        print(f"[Client] Connection failed: {e}. Retrying in {RECONNECT_DELAY}s...")
                    time.sleep(RECONNECT_DELAY)
            else:
                time.sleep(0.5)

    def _receive_loop(self):
        """Continuously receive messages from the server."""
        while True:
            with self._lock:
                sock = self.sock if self.connected else None

            if sock is None:
                time.sleep(0.05)
                continue

            try:
                chunk = sock.recv(1024)
                if not chunk:
                    self._disconnect()
                    continue
                self._recv_buffer += chunk.decode('utf-8')

                while '\n' in self._recv_buffer:
                    line, self._recv_buffer = self._recv_buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            self._handle_server_message(msg)
                        except json.JSONDecodeError as e:
                            if not self.silent:
                                print(f"[Client] Bad JSON from server: {e}")

            except socket.timeout:
                pass  # Normal, no data
            except (ConnectionResetError, OSError):
                self._disconnect()

    def _handle_server_message(self, msg):
        """Process a message received from the server."""
        cmd = msg.get("cmd")
        if cmd == "go":
            if not self.silent:
                print("[Client] Received 'go' from server!")
            if self.go_callback:
                self.go_callback()
        else:
            if not self.silent:
                print(f"[Client] Unknown server message: {msg}")

    def _disconnect(self):
        with self._lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = None
            self.connected = False
        if not self.silent:
            print("[Client] Disconnected from server.")

    def sendStatus(self,
                bounding_boxes,     # [[x,y,w,h], [x,y,w,h], ...]  <-- changed
                velocity,
                search_location,
                search_size,
                frame_size,
                thumbnail = None):

        if not self.connected:
            return

        payload = {
            "bounding_boxes":  list(bounding_boxes),  # <-- changed
            "velocity":        list(velocity),
            "search_location": list(search_location),
            "search_size":     list(search_size),
            "frame_size":      list(frame_size)
        }
        # Encode image if provided
        if thumbnail is not None:
            img = np.asarray(thumbnail, dtype=np.uint8)
            payload["thumbnail_shape"] = list(img.shape)          # e.g. [128, 128]
            payload["thumbnail_data"]  = base64.b64encode(
                                                img.tobytes()
                                            ).decode('ascii')
        message = json.dumps(payload) + "\n"

        with self._lock:
            sock = self.sock

        if sock is None:
            return

        try:
            sock.sendall(message.encode('utf-8'))
        except (BrokenPipeError, OSError):
            self._disconnect()
            
    def _send_handshake(self):
        """Send client ID immediately after connecting."""
        handshake = json.dumps({"handshake": True, "client_id": self.client_id}) + "\n"
        self.sock.sendall(handshake.encode('utf-8'))


# ---------------------------------------------------------------------------
# Demo / test main loop
# ---------------------------------------------------------------------------
def on_go():
    """Called when the server sends a 'go' command."""
    
    print("[Client] *** GO command received! Executing action... ***")


def main():
    client = StatusClient(SERVER_HOST, SERVER_PORT)
    client.set_go_callback(on_go)

    # Wait a moment for connection to establish
    time.sleep(1.0)

    frame_w, frame_h = 640, 480
    search_w, search_h = 160, 120

    # Simulate a moving target
    target_x = frame_w / 2
    target_y = frame_h / 2
    vx = random.uniform(30, 80)   # pixels/sec
    vy = random.uniform(30, 80)
    bb_w, bb_h = 60, 40

    if not self.silent:
        print("[Client] Sending status updates. Press Ctrl+C to stop.")
    try:
        while True:
            dt = 1.0 / 30.0  # 30 Hz

            # Move target, bounce off walls
            target_x += vx * dt
            target_y += vy * dt
            if target_x < 0 or target_x > frame_w:
                vx = -vx
                target_x = max(0, min(frame_w, target_x))
            if target_y < 0 or target_y > frame_h:
                vy = -vy
                target_y = max(0, min(frame_h, target_y))
            
            bounding_boxes  = [[target_x - bb_w/2, target_y - bb_h/2, bb_w, bb_h]]  # <-- list of one
            velocity        = [vx, vy]
            search_location = [target_x, target_y]   # center search on target
            search_size     = [search_w, search_h]
            frame_size      = [frame_w, frame_h]
            thumbnail = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

            client.sendStatus(bounding_boxes, velocity,
                              search_location, search_size, frame_size, thumbnail)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("[Client] Stopped.")


if __name__ == "__main__":
    main()
