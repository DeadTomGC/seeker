USE_VIDEO_FILE = False
try:
    import picamera2
except:
    print("no picam, hope you're good with that... ")
    USE_VIDEO_FILE = True
import time
import sys
import cv2
from client import *
from seeker import Seeker
from pico_link import PicoLink, steer_to_location, CTRL_FLAG_SEEK_ACTIVE, TRACKING_NONE, TRACKING_LOCKED

pico = None
go_active = False

def show_input(img):
    cv2.imshow('input', img)
    cv2.waitKey(1)

def show_debug(img):
    cv2.imshow('debug', img)
    cv2.waitKey(1)

def on_go():
    """GO notification: from the Pico (SEEK engaged on the RC switch) or a
    server 'go' click.  Motor arming is decided by the Pico from the RC."""
    global go_active
    go_active = True
    print("[Seeker] GO: pursuit engaged (motor arming is on the RC switch)")

def draw_test_time_and_count_info(img,frame_number=0,time_ratio = 0,bg_metric=0):
    cv2.putText(img, f"Frame: {frame_number}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"Time Ratio: {time_ratio:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"Total Areas: {bg_metric}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

video_file = ''
# Check for --video command line argument
if '--video' in sys.argv:
    USE_VIDEO_FILE = True
    video_idx = sys.argv.index('--video')
    if video_idx + 1 < len(sys.argv):
        video_file = sys.argv[video_idx + 1]

if '--host' in sys.argv:
    host_idx = sys.argv.index('--host')
    if host_idx + 1 < len(sys.argv):
        SERVER_HOST = sys.argv[host_idx + 1]


if video_file == '' and USE_VIDEO_FILE:
    video_file = 'short_video.mp4'
    

if USE_VIDEO_FILE:
    print(f"Using video file: {video_file}")
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    frame_size = (image.shape[1], image.shape[0]) if success else (640, 480)
    frame_rate = 30
else:
    picam = picamera2.Picamera2()
    frame_size = (640, 480)

    modes = picam.sensor_modes

    for mode in modes:
        # go for largest Y value that can hit 30fps
        if frame_size[1] <= mode["size"][1] and mode["size"][0] < 1920 and mode["fps"] >= 30:
            frame_size = mode["size"]
            frame_rate = mode["fps"]
            print(f"{frame_size},{frame_rate}")

    print(f"Selected Mode = size:{frame_size},fps:{frame_rate}")

    config = picam.create_video_configuration(
        {"size": frame_size, "format": "YUV420"},  # Main stream size
        controls={"FrameRate": frame_rate}
    )
    picam.configure(config)
    picam.options["compress_level"] = 0
    picam.start()

    image = picam.capture_array()
    image = image[:frame_size[1], :frame_size[0]]
    success = True

count = 0

if not USE_VIDEO_FILE:
    client = StatusClient(SERVER_HOST, SERVER_PORT)
    client.set_go_callback(on_go)

    try:
        pico = PicoLink(silent=False)
        print("[Seeker] Pico link running")
    except Exception as e:
        print(f"[Seeker] Pico link unavailable: {e}")

image_format = 'BGR' if USE_VIDEO_FILE else 'YUV420'
tracker = Seeker(frame_size, image_format=image_format)

if not USE_VIDEO_FILE:
    time.sleep(1.0)

start_time = time.time() * 1000

video_name = video_file if USE_VIDEO_FILE else 'Pi Camera'

if success:
    print(f"Video Opened: {video_name}")
else:
    print("Error, Exiting")

paused = True if USE_VIDEO_FILE else False

#pre scroll to frame of interest, if the user has specified a start frame
if USE_VIDEO_FILE:
    startFrame = 2080
    while count < startFrame:
        success, image = vidcap.read()
        count += 1

while success:
    if USE_VIDEO_FILE:
        g_time = count * 0.033
    else:
        g_time = time.time() * 1000

    tracker.process_frame(image, g_time)

    status = tracker.get_status()
    search_location = status['search_location']
    velocity = status['velocity']
    bounding_boxes = status['bounding_boxes']
    count = status['count']

    # The Pico commands the RPi's state over the link; like the control
    # frames, this state rides in every telemetry frame so missed messages
    # self-heal.  Standby continually resets tracks so no stale track is
    # revived when control returns to automatic; GO fires once per edge.
    if pico is not None:
        telem = pico.get_telemetry()
        if telem is not None and telem['age_s'] < 1.0:
            if telem['pi_standby']:
                tracker.reset_track()
                status = tracker.get_status()
                search_location = status['search_location']
                velocity = status['velocity']
                bounding_boxes = status['bounding_boxes']
            if telem['pi_go'] and not go_active:
                on_go()
            go_active = telem['pi_go']

    if pico is not None:
        if status['tracking']:
            seek_active, roll_cmd, pitch_cmd, yaw_cmd = steer_to_location(
                status['search_location'], frame_size)
            flags = CTRL_FLAG_SEEK_ACTIVE
        else:
            seek_active, roll_cmd, pitch_cmd, yaw_cmd = False, 0.0, 0.0, 0.0
            flags = 0
        # Tracker state rides in every control frame: missed messages
        # self-heal and the Pico eventually sees any status change.
        pico.set_command(flags,
                         TRACKING_LOCKED if status['tracking'] else TRACKING_NONE,
                         roll_cmd, pitch_cmd, yaw_cmd)

    end_time = time.time() * 1000

    if not USE_VIDEO_FILE:
        tiny_grey = tracker.get_tiny_grey_image()
        if client.connected:
            client.sendStatus(bounding_boxes, velocity,
                              search_location, [status['search_size'], status['search_size']],
                              frame_size, tiny_grey)

    if count % 10 == 0:
        print(f"\rConnection: {'N/A (video file)' if USE_VIDEO_FILE else client.connected} "
              f"Location: {search_location} Velocity: {velocity[0]:.2f},{velocity[1]:.2f} "
              f"Frame Time: {(end_time - start_time):.2f}ms, Tiny Time {tracker.tiny_time:.2f}", end="")
        if pico is not None:
            telem = pico.get_telemetry()
            if telem is not None:
                print(f"\n[Pico] status:0x{telem['status']:02X} "
                      f"standby:{telem['pi_standby']} go:{telem['pi_go']} "
                      f"armed:{telem['motor_armed']} euler:{telem['euler']} "
                      f"battery:{telem['battery_v']:.1f}V age:{telem['age_s']:.1f}s")
    if end_time - start_time > 40:
        pass
        #print(f"\nHigh Frame Time: {end_time - start_time}ms")
        
    if USE_VIDEO_FILE:
        debug_frame = tracker.get_debug_bgr_image(image)
        draw_test_time_and_count_info(debug_frame, count, tracker.time_ratio,bg_metric=tracker.bg_metric)
        show_input(debug_frame)
        show_debug(cv2.resize(tracker.processed_grey, (256, 256), interpolation=cv2.INTER_NEAREST))
        #if the space bar is pressed, pause the loop until the space bar is pressed again
        #if, while the loop is paused, the user presses the n key, continue to the next frame and pause on that frame
        key = cv2.waitKey(25)& 0xFF
        if paused or key == ord(' '):

            paused = True
            print("\nPaused. Press space to continue, n for next frame.")
            while True:
                key2 = cv2.waitKey(25) & 0xFF
                if key2 == ord(' '):
                    paused = False
                    print("Continuing...")
                    break
                elif key2 == ord('n'):
                    print("Next frame...")
                    break
                if key2 == ord('r'):
                    print("Resetting tracker...")
                    tracker.reset_track()
                    break
    start_time = time.time() * 1000

    if USE_VIDEO_FILE:
        success, image = vidcap.read()
    else:
        image = picam.capture_array()
        image = image[:frame_size[1], :frame_size[0]]

if not USE_VIDEO_FILE:
    picam.stop()