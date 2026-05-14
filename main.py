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


def show_input(img):
    cv2.imshow('input', img)
    cv2.waitKey(1)

def show_debug(img):
    cv2.imshow('debug', img)
    cv2.waitKey(1)

g_time = 0

video_file = ''
# Check for --video command line argument
if '--video' in sys.argv:
    USE_VIDEO_FILE = True
    video_idx = sys.argv.index('--video')
    if video_idx + 1 < len(sys.argv):
        video_file = sys.argv[video_idx + 1]


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

    end_time = time.time() * 1000

    if not USE_VIDEO_FILE:
        tiny_grey = tracker.get_tiny_grey_image()
        if client.connected:
            client.sendStatus(bounding_boxes, velocity,
                              status['search_center'], [status['search_size'], status['search_size']],
                              frame_size, tiny_grey)

    if count % 10 == 0:
        print(f"\rConnection: {'N/A (video file)' if USE_VIDEO_FILE else client.connected} "
              f"Location: {search_location} Velocity: {velocity[0]:.2f},{velocity[1]:.2f} "
              f"Frame Time: {(end_time - start_time):.2f}ms, Tiny Time {tracker.tiny_time:.2f}", end="")
    if end_time - start_time > 40:
        print(f"\nHigh Frame Time: {end_time - start_time}ms")
        
    if USE_VIDEO_FILE:
        debug_frame = tracker.get_debug_bgr_image(image)
        show_input(debug_frame)
        show_debug(tracker.big_proc_extra_tiny)
    start_time = time.time() * 1000

    if USE_VIDEO_FILE:
        success, image = vidcap.read()
    else:
        image = picam.capture_array()
        image = image[:frame_size[1], :frame_size[0]]

if not USE_VIDEO_FILE:
    picam.stop()