import socket
import threading
import json
import pygame
import sys
from collections import defaultdict
import base64
import numpy as np
import time

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 65432
MAX_CLIENTS = 4
WINDOW_SIZE = (1280, 960)
BG_COLOR = (30, 30, 30)
FRAME_BG_COLOR = (50, 50, 50)
SEARCH_BOX_COLOR = (0, 180, 255)
BOUNDING_BOX_COLOR = (0, 255, 0)
FRAME_BORDER_COLOR = (180, 180, 180)
GRID_PADDING = 20

# --- Shared State ---
go_queue = {}         # client_id -> threading.Event or flag
client_registry = {}
registry_lock   = threading.Lock()

RECONNECT_GRACE_PERIOD = 30.0   # seconds to hold a slot after disconnect


def get_grid_layout(num_clients):
    """Return (cols, rows) for grid layout based on number of clients."""
    if num_clients <= 1:
        return (1, 1)
    elif num_clients <= 2:
        return (2, 1)
    elif num_clients <= 4:
        return (2, 2)
    return (2, 2)


def get_frame_rect(client_index, num_clients, window_w, window_h):
    """Return pygame.Rect for where a client's frame should be drawn."""
    cols, rows = get_grid_layout(num_clients)
    cell_w = (window_w - GRID_PADDING * (cols + 1)) // cols
    cell_h = (window_h - GRID_PADDING * (rows + 1)) // rows
    col = client_index % cols
    row = client_index // cols
    x = GRID_PADDING + col * (cell_w + GRID_PADDING)
    y = GRID_PADDING + row * (cell_h + GRID_PADDING)
    return pygame.Rect(x, y, cell_w, cell_h)


def get_or_assign_slot(client_id):
    """Return existing slot for client_id, or assign a new one. Returns -1 if full."""
    with registry_lock:
        # Reuse existing slot (connected or recently disconnected)
        if client_id in client_registry:
            return client_registry[client_id]['slot']

        # Expire stale disconnected clients beyond grace period
        now = time.time()
        for cid, info in list(client_registry.items()):
            if (not info['connected'] and
                info['disconnect_time'] is not None and
                now - info['disconnect_time'] > RECONNECT_GRACE_PERIOD):
                print(f"[Server] Expiring stale slot {info['slot']} for {cid}")
                del client_registry[cid]

        # Find a free slot (0-3)
        used_slots = {info['slot'] for info in client_registry.values()}
        for slot in range(4):
            if slot not in used_slots:
                client_registry[client_id] = {
                    'slot':            slot,
                    'data':            None,
                    'conn':            None,
                    'addr':            None,
                    'connected':       False,
                    'disconnect_time': None
                }
                return slot

        return -1   # no free slots


def handle_client(conn, addr):
    print(f"[Server] Connection from {addr}")
    client_id = None

    try:
        buffer = ""
        while "\n" not in buffer:
            chunk = conn.recv(4096).decode('utf-8')
            if not chunk:
                return
            buffer += chunk

        line, buffer = buffer.split("\n", 1)
        msg = json.loads(line)

        if not msg.get("handshake") or "client_id" not in msg:
            print(f"[Server] No handshake from {addr}, dropping.")
            conn.close()
            return

        client_id = msg["client_id"]
        slot = get_or_assign_slot(client_id)

        if slot == -1:
            print(f"[Server] No free slots for {client_id}, dropping.")
            conn.close()
            return

        with registry_lock:
            client_registry[client_id].update({
                'conn':            conn,
                'addr':            addr,
                'connected':       True,
                'disconnect_time': None
            })

        print(f"[Server] Client {client_id} assigned to slot {slot}")

        while True:
            while "\n" not in buffer:
                chunk = conn.recv(65536).decode('utf-8')
                if not chunk:
                    return
                buffer += chunk

            line, buffer = buffer.split("\n", 1)
            data = json.loads(line)

            with registry_lock:
                client_registry[client_id]['data'] = data

    except Exception as e:
        print(f"[Server] Client {client_id or addr} error: {e}")

    finally:
        conn.close()
        if client_id and client_id in client_registry:
            with registry_lock:
                # Only mark disconnected if this thread's connection is still
                # the one registered. If a new thread has already reconnected,
                # client_registry will hold the NEW conn object, not ours.
                if client_registry[client_id].get('conn') is conn:
                    client_registry[client_id]['connected']       = False
                    client_registry[client_id]['disconnect_time'] = time.time()
                    client_registry[client_id]['conn']            = None
                    print(f"[Server] Client {client_id} disconnected, holding slot {slot} "
                          f"for {RECONNECT_GRACE_PERIOD}s")
                else:
                    print(f"[Server] Client {client_id} old thread exiting, "
                          f"new connection already active — skipping disconnect.")

def decode_thumbnail(data):
    """Decode base64 thumbnail from status message. Returns pygame.Surface or None."""
    if "thumbnail_data" not in data or "thumbnail_shape" not in data:
        return None
    try:
        shape = data["thumbnail_shape"]               # [h, w] or [h, w, c]
        raw   = base64.b64decode(data["thumbnail_data"])
        img   = np.frombuffer(raw, dtype=np.uint8).reshape(shape)

        # Convert greyscale [H,W] to RGB [H,W,3] for pygame
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        # pygame.surfarray expects (width, height, 3) with x-major order
        img_transposed = np.transpose(img, (1, 0, 2))
        return pygame.surfarray.make_surface(img_transposed)
    except Exception as e:
        print(f"[Server] Failed to decode thumbnail: {e}")
        return None

def draw_client_frame(surface, rect, data, client_id, font, connected=True):  # <-- add connected param
    # Background
    bg_color = FRAME_BG_COLOR if connected else (120, 0, 0)  # <-- red when disconnected
    pygame.draw.rect(surface, bg_color, rect)
    pygame.draw.rect(surface, FRAME_BORDER_COLOR, rect, 2)


    if data is None:
        # No data yet
        label = font.render(f"Client {client_id} - waiting...", True, (200, 200, 200))
        surface.blit(label, (rect.x + 10, rect.y + 10))
        return

    try:
        frame_w = data['frame_size'][0]
        frame_h = data['frame_size'][1]
        search_x = data['search_location'][0]
        search_y = data['search_location'][1]
        search_w = data['search_size'][0]
        search_h = data['search_size'][1]
        bb = data['bounding_boxes']   # <-- changed, now a list
    except (KeyError, TypeError) as e:
        label = font.render(f"Client {client_id} - bad data", True, (255, 80, 80))
        surface.blit(label, (rect.x + 10, rect.y + 10))
        return

    if frame_w <= 0 or frame_h <= 0:
        return

    # Available drawing area inside the cell (with inner padding)
    inner_pad = 30
    draw_x = rect.x + inner_pad
    draw_y = rect.y + inner_pad + 20  # leave room for label at top
    draw_w = rect.width - inner_pad * 2
    draw_h = rect.height - inner_pad * 2 - 20

    # Scale factors: map frame coordinates to cell drawing area
    scale_x = draw_w / frame_w
    scale_y = draw_h / frame_h

    # Draw the frame boundary
    frame_rect = pygame.Rect(draw_x, draw_y, draw_w, draw_h)
    pygame.draw.rect(surface, (80, 80, 80), frame_rect)
    pygame.draw.rect(surface, FRAME_BORDER_COLOR, frame_rect, 1)

    # Draw search box (centered at search_location with search_size)
    s_px = draw_x + int((search_x - search_w / 2) * scale_x)
    s_py = draw_y + int((search_y - search_h / 2) * scale_y)
    s_pw = int(search_w * scale_x)
    s_ph = int(search_h * scale_y)
    search_rect = pygame.Rect(s_px, s_py, s_pw, s_ph)
    search_rect.clip(frame_rect)  # don't draw outside frame
    

    # Client label
    label = font.render(f"Client {client_id}", True, (220, 220, 220))
    surface.blit(label, (rect.x + 8, rect.y + 6))
    
    # Thumbnail — draw in top-left corner of the cell if present
    thumb_surface = decode_thumbnail(data)
    if thumb_surface is not None:
        # Scale to exactly fit the search box in display coordinates
        scaled_thumb = pygame.transform.scale(thumb_surface, (s_pw, s_ph))
        surface.blit(scaled_thumb, (s_px, s_py))
        pygame.draw.rect(surface, SEARCH_BOX_COLOR,
                     pygame.Rect(s_px, s_py, s_pw, s_ph), 2)
    
    pygame.draw.rect(surface, SEARCH_BOX_COLOR, search_rect, 2)
    
    # Draw bounding box (position is relative to overall frame)
    for bb in data['bounding_boxes']:                          # <-- loop added
        bb_x, bb_y, bb_w, bb_h = (float(v) for v in bb)
        b_px = draw_x + int(bb_x * scale_x)
        b_py = draw_y + int(bb_y * scale_y)
        b_pw = max(1, int(bb_w * scale_x))
        b_ph = max(1, int(bb_h * scale_y))
        pygame.draw.rect(surface, BOUNDING_BOX_COLOR,
                         pygame.Rect(b_px, b_py, b_pw, b_ph), 2)
        
    


def server_accept_loop(server_sock):
    """Continuously accept new client connections."""
    while True:
        try:
            conn, addr = server_sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            print(f"[Server] Accept error: {e}")
            break


def get_active_slots():
    """Return dict of slot -> data for connected clients (or recently disconnected)."""
    with registry_lock:
        result = {}
        for info in client_registry.values():
            if info['data'] is not None:
                result[info['slot']] = info
        return result


def run_server():
    # Start TCP server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(MAX_CLIENTS)
    print(f"[Server] Listening on {HOST}:{PORT}")

    accept_thread = threading.Thread(target=server_accept_loop, args=(server_sock,), daemon=True)
    accept_thread.start()

    # Pygame display
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Multi-Client Status Monitor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    running = True
    while running:
        win_w, win_h = screen.get_size()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                with registry_lock:
                    # Build a slot-sorted snapshot for hit testing
                    sorted_entries = sorted(
                        client_registry.items(),
                        key=lambda kv: kv[1]['slot']
                    )
                num = len(sorted_entries)
                if num == 0:
                    continue
                for i, (cid, info) in enumerate(sorted_entries):
                    r = get_frame_rect(i, num, win_w, win_h)
                    if r.collidepoint(mx, my):
                        with registry_lock:
                            go_queue[cid] = True
                        print(f"[Server] 'go' queued for client {cid}")

        screen.fill(BG_COLOR)

        with registry_lock:
            now = time.time()
            # Sort by slot so layout is stable
            sorted_entries = sorted(
                [
                    (cid, info) for cid, info in client_registry.items()
                    if info['connected'] or                              # <-- still connected
                       info['disconnect_time'] is None or               # <-- never disconnected
                       (now - info['disconnect_time']) <= RECONNECT_GRACE_PERIOD  # <-- within grace
                ],
                key=lambda kv: kv[1]['slot']
            )

        num = len(sorted_entries)
        if num == 0:
            msg = font.render("Waiting for clients...", True, (180, 180, 180))
            screen.blit(msg, (win_w // 2 - msg.get_width() // 2,
                               win_h // 2 - msg.get_height() // 2))
        else:
            for i, (cid, info) in enumerate(sorted_entries):
                r = get_frame_rect(i, num, win_w, win_h)
                draw_client_frame(screen, r, info['data'], cid, font, info['connected'])


        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    server_sock.close()
    sys.exit(0)



if __name__ == "__main__":
    run_server()
