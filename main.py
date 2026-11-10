USE_VIDEO_FILE = False
try:
    import picamera2
except:
    print("no picam, hope you're good with that... ")
    USE_VIDEO_FILE = True
import time
import math
import sys
import cv2
from client import *
from seeker import Seeker
from pico_link import PicoLink, CTRL_FLAG_SEEK_ACTIVE, TRACKING_NONE, TRACKING_LOCKED

pico = None
go_active = False

# ---------------------------------------------------------------------------
# Pursuit control: null the track's true radial (angular) velocity
# ---------------------------------------------------------------------------
# Tunables: diagonal field of view of the camera/lens and the PID gains.
CAMERA_DIAGONAL_FOV_DEG = 65.0

# (Kp, Ki, Kd) mapping deg/s of true track velocity to a -100..100 rate cmd.
YAW_PID_GAINS = (3.0, 0.5, 0.2)
PITCH_PID_GAINS = (3.0, 0.5, 0.2)
RADIAL_DEADBAND_DPS = 0.5          # deg/s: below this the axis counts as nulled

# (Kp, Ki, Kd) mapping the centring error in deg to a -100..100 rate cmd.
YAW_CENTER_GAINS = (5.0, 0.2, 1.0)
PITCH_CENTER_GAINS = (5.0, 0.2, 1.0)
CENTER_DEADBAND_PX = 10.0          # px from centre: inside this, no steering
STEERING_FF_GAIN = 0.8             # feed-forward from true track velocity
MAX_STEER_CMD = 100.0
MAX_STEER_DT = 0.1                 # s: control-loop dt clamp

# Sign fixes for the IMU mounting; flip to -1.0 per airframe if the axis
# responds the wrong way (mirrors the sign caveat in steer_to_location).
GYRO_YAW_SIGN = 1.0                # gz: positive = nose right
GYRO_PITCH_SIGN = 1.0              # gx: positive = nose up

# The tracker's polyfit velocity is px/ms (fit over millisecond timestamps).
TRACKER_VELOCITY_PX_S_SCALE = 1000.0


def deg_per_pixel(frame_size, diagonal_fov_deg=CAMERA_DIAGONAL_FOV_DEG):
    """Angular resolution of the image given the diagonal FOV."""
    return diagonal_fov_deg / math.hypot(frame_size[0], frame_size[1])


def pixels_to_deg_per_s(velocity_px_s, frame_size,
                        diagonal_fov_deg=CAMERA_DIAGONAL_FOV_DEG):
    """Convert a track velocity in px/s into deg/s about the X (pitch) and
    Z (yaw) axes, using the diagonal FOV and image resolution."""
    deg_per_px = deg_per_pixel(frame_size, diagonal_fov_deg)
    return velocity_px_s[0] * deg_per_px, velocity_px_s[1] * deg_per_px


class PID:
    """Tunable PID: input an error, output a steering command clamped to
    -100..100.  The integral is clamped and the deadband zeroes the error
    around the setpoint.  The derivative is skipped on the first update
    after a reset to avoid a startup spike."""

    def __init__(self, gains, max_cmd=MAX_STEER_CMD, deadband=0.0):
        self.kp, self.ki, self.kd = gains
        self.max_cmd = max_cmd
        self.deadband = deadband
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self._first = True

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0
        if abs(error) < self.deadband:
            error = 0.0
        self.integral += error * dt
        i_limit = self.max_cmd / max(self.ki, 1e-6)
        self.integral = max(-i_limit, min(i_limit, self.integral))
        deriv = 0.0 if self._first else (error - self.prev_error) / dt
        self._first = False
        self.prev_error = error
        cmd = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(-self.max_cmd, min(self.max_cmd, cmd))


# Velocity-nulling PIDs (pursuit): input true track velocity in deg/s.
yaw_rate_pid = PID(YAW_PID_GAINS, deadband=RADIAL_DEADBAND_DPS)
pitch_rate_pid = PID(PITCH_PID_GAINS, deadband=RADIAL_DEADBAND_DPS)
# Centering PIDs (seek): input angular offset from centre in deg.
yaw_center_pid = PID(YAW_CENTER_GAINS)
pitch_center_pid = PID(PITCH_CENTER_GAINS)
last_ctrl_time = time.time()


def true_track_velocity_dps(velocity_px_ms, frame_size, gyro_dps):
    """True angular velocity of the track in deg/s: the apparent image
    velocity plus our own rotation rate (they cancel when the target is
    stationary in the world and we are turning)."""
    vx_dps, vy_dps = pixels_to_deg_per_s(
        (velocity_px_ms[0] * TRACKER_VELOCITY_PX_S_SCALE,
         velocity_px_ms[1] * TRACKER_VELOCITY_PX_S_SCALE), frame_size)
    # Z faces forward, so yaw rate is gz and pitch rate is gx.
    return (vx_dps + GYRO_YAW_SIGN * gyro_dps[2],
            vy_dps + GYRO_PITCH_SIGN * gyro_dps[0])


def null_radial_velocity(velocity_px_ms, frame_size, gyro_dps, dt,
                         yaw_ctrl=yaw_rate_pid, pitch_ctrl=pitch_rate_pid):
    """Steer to null the track's true radial velocity, compensating for our
    own rotation rate.

    Returns (pitch_cmd, yaw_cmd) in -100..100.  Sign conventions match
    steer_to_location: track moving left (true vx < 0) -> steer left."""
    true_vx, true_vy = true_track_velocity_dps(velocity_px_ms, frame_size, gyro_dps)
    yaw_cmd = yaw_ctrl.update(true_vx, dt)
    pitch_cmd = pitch_ctrl.update(true_vy, dt)
    return pitch_cmd, yaw_cmd


def steer_to_location(search_location, frame_size, velocity_px_ms, gyro_dps,
                      dt, yaw_ctrl=yaw_center_pid, pitch_ctrl=pitch_center_pid):
    """Steer to hold the tracked location centred in the camera's view.

    Same tunable-PID scheme as null_radial_velocity, but the error is the
    angular offset of the search location from the image centre.  A
    feed-forward term from the track's true (rotation-compensated) velocity
    pre-positions the nose for a moving track.  Returns
    (seek_active, roll, pitch, yaw) rate commands in -100..100, with roll
    always 0 (the RPi does not steer roll; the pilot does).

    Signs are set for a conventional layout and may need flipping per
    airframe (see GYRO_*_SIGN): target right of centre -> yaw right
    (positive); below centre -> pitch positive."""
    offset_x = search_location[0] - frame_size[0] / 2.0
    offset_y = search_location[1] - frame_size[1] / 2.0

    if abs(offset_x) < CENTER_DEADBAND_PX and abs(offset_y) < CENTER_DEADBAND_PX:
        yaw_ctrl.reset()
        pitch_ctrl.reset()
        return False, 0.0, 0.0, 0.0

    deg_per_px = deg_per_pixel(frame_size)
    true_vx, true_vy = true_track_velocity_dps(velocity_px_ms, frame_size, gyro_dps)

    yaw_cmd = yaw_ctrl.update(offset_x * deg_per_px, dt) + STEERING_FF_GAIN * true_vx
    pitch_cmd = pitch_ctrl.update(offset_y * deg_per_px, dt) + STEERING_FF_GAIN * true_vy
    yaw_cmd = max(-MAX_STEER_CMD, min(MAX_STEER_CMD, yaw_cmd))
    pitch_cmd = max(-MAX_STEER_CMD, min(MAX_STEER_CMD, pitch_cmd))
    return True, 0.0, pitch_cmd, yaw_cmd

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
        now_ctrl = time.time()
        dt = min(max(now_ctrl - last_ctrl_time, 1e-3), MAX_STEER_DT)
        last_ctrl_time = now_ctrl
        if status['tracking']:
            gyro = telem['gyro'] if (telem is not None and telem['age_s'] < 1.0) \
                else (0.0, 0.0, 0.0)
            if go_active:
                pitch_cmd, yaw_cmd = null_radial_velocity(
                    velocity, frame_size, gyro, dt)
                seek_active, roll_cmd = True, 0.0
                yaw_center_pid.reset()
                pitch_center_pid.reset()
            else:
                yaw_rate_pid.reset()
                pitch_rate_pid.reset()
                seek_active, roll_cmd, pitch_cmd, yaw_cmd = steer_to_location(
                    status['search_location'], frame_size, velocity, gyro, dt)
            flags = CTRL_FLAG_SEEK_ACTIVE
        else:
            yaw_rate_pid.reset()
            pitch_rate_pid.reset()
            yaw_center_pid.reset()
            pitch_center_pid.reset()
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