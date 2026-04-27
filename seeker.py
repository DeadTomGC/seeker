


import cv2
import numpy as np
import time
import math
import picamera2

g_time = 0

def show_input(img):
    cv2.imshow('input', img)
    cv2.waitKey(1)

def show_processed(img):
    cv2.imshow('processed', img)
    cv2.waitKey(1)

def crop_square(img, center, size):
    h, w = img.shape[:2]
    half = size // 2
    
    x = max(half, min(center[0], w - half))
    y = max(half, min(center[1], h - half))
    
    return img[y - half:y + half, x - half:x + half], (x, y)

def to_greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def yuv_to_greyscale(img):
    h, w = img.shape[:2]
    return img[:h, :w]

def resize_image(img, size):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    scale = size / w, size / h
    return resized, scale

def rescale_image(img, scale):
    h, w = img.shape[:2]
    rescaled= cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    return rescaled

def to_cartesian(img, coord):
    h, w = img.shape[:2]
    return (int(coord[0] - w // 2), int(h // 2 - coord[1]))

def to_image(img, coord):
    h, w = img.shape[:2]
    return (coord[0] + w // 2, h // 2 - coord[1])


def edge_canny(img, low=50, high=200):
    return cv2.Canny(img, low, high)

def to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def find_filter_closed_contours(edge_img,kernel_size = 7,min_solidity = 1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ih, iw = edge_img.shape[:2]
    good_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        #hull_area = cv2.contourArea(cv2.convexHull(c))
        perimeter = cv2.arcLength(c,False)
        x, y, cw, ch = cv2.boundingRect(c)
        #if hull_area > 0 and area / hull_area >= min_solidity:
        if perimeter>0 and area/perimeter >= min_solidity and cw/iw < 0.8 and ch/ih < 0.8:#*area/100:
            good_contours.append(c)
    return good_contours

def draw_bounding_boxes(img, contours, scale, color=(0, 255, 0), thickness=2,search_center = None,search_size = None):
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = int(x/float(scale)), int(y/float(scale)), int(w/float(scale)), int(h/float(scale))
        #if search_center and search_size:
        image_search_center = to_image(img,search_center)
        x = x + (image_search_center[0]-search_size//2)
        y = y + (image_search_center[1]-search_size//2)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

def draw_search_area(img,search_center,search_size,color = (0,0,255),thickness = 2):
    image_search_center = to_image(img,search_center)
    x = image_search_center[0]-search_size//2
    y = image_search_center[1]-search_size//2
    w = search_size
    h = search_size
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

def draw_contours(img, contours, color=(0, 255, 0), thickness=2):
    cv2.drawContours(img, contours, -1, color, thickness)

class trackDataPoint:
    def __init__(self, center,area):
        self.area = area
        self.timestamp = time.time() * 1000
        self.cx, self.cy = center[0],center[1]

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0, 0
    return M["m10"] / M["m00"], M["m01"] / M["m00"]

def track_contour(contours: list, past_data: list, max_data_length=5,missed_tracks = 0):
    velocity = (0, 0)

    if not past_data:
        return past_data, velocity, False

    n = len(past_data)

    # estimate expected position and area from linear fit or last point
    if n >= 2:
        times = np.array([p.timestamp for p in past_data])
        xs = np.array([p.cx for p in past_data])
        ys = np.array([p.cy for p in past_data])
        areas = np.array([p.area for p in past_data])

        t = times - times[0]
        vx = np.polyfit(t, xs, 1)[0]
        vy = np.polyfit(t, ys, 1)[0]
        velocity = (vx, vy)

        dt = time.time() * 1000 - past_data[-1].timestamp
        exp_x = past_data[-1].cx + vx * dt
        exp_y = past_data[-1].cy + vy * dt
        exp_area = np.mean(areas)
    else:
        exp_x = past_data[-1].cx
        exp_y = past_data[-1].cy
        exp_area = past_data[-1].area

    # larger margin when little data
    margin_scale = max(1.0, 3.0 / n)
    margin_scale = margin_scale + (0.1*missed_tracks)
    pos_margin = 70 * margin_scale
    area_margin = 2 * margin_scale  # fraction of expected area

    # find best matching contour
    best = None
    best_dist = float('inf')
    for c in contours:
        cx, cy = c.cx,c.cy
        area = c.area
        dist = np.hypot(cx - exp_x, cy - exp_y)
        area_diff = abs(area - exp_area) / (exp_area + 1e-5)
        if dist < pos_margin and area_diff < area_margin and dist < best_dist:
            best_dist = dist
            best = c

    if best is None:
        return past_data, velocity, False

    new_data = past_data + [best]
    if len(new_data) > max_data_length:
        new_data = new_data[-max_data_length:]

    #recalculate velocity
    times = np.array([p.timestamp for p in new_data])
    xs = np.array([p.cx for p in new_data])
    ys = np.array([p.cy for p in new_data])
    areas = np.array([p.area for p in new_data])

    t = times - times[0]
    vx = np.polyfit(t, xs, 1)[0]
    vy = np.polyfit(t, ys, 1)[0]
    new_velocity = (vx, vy)
    
    return new_data, new_velocity, True

def max_bounding_box_length(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return max(w,h)

def contours_to_trackData(contours):
    return [trackDataPoint(get_contour_center(c),max_bounding_box_length(c)) for c in contours]

def scale_move_to_global_image(img,scale,center,size,trackData):
    frame_center = to_image(img,center)
    crop_top_left = (frame_center[0]-size//2,frame_center[1]-size//2)
    return [trackDataPoint((p.cx/scale+crop_top_left[0],p.cy/scale+crop_top_left[1]),p.area/scale) for p in trackData]

def point_distance(p1, p2):
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])


#video_name = 'short_video.mp4'
#
#vidcap = cv2.VideoCapture(video_name)
#success, image = vidcap.read()

picam = picamera2.Picamera2()
frame_size = (1280,960)
config = picam.create_video_configuration(
    {"size": frame_size,"format": "YUV420"},  # Main stream size
    #lores={"size": (1280, 960),"format": "YUV420"},  # Low-res stream for fast access
    controls={"FrameRate": 43}
)
picam.configure(config)
picam.options["compress_level"] = 0
picam.start()
video_name = 'Pi Camera'
image = picam.capture_array()
image = image[:frame_size[1],:frame_size[0]]
#image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
success = True
count = 0

entity_detected = False
entity_location = (0,0)
search_size = 400
search_location = (0,0)
smaller_size = 128
video_track_data = []
not_found_count = 0
slow_down = False
velocity = (0,0)
start_time = time.time()*1000
if success:
    print(f"Video Opened: {video_name}")
else:
    print("Error, Exiting")
while success:
    #TODO process image
    
    cropped_grey,new_location = crop_square(image,to_image(image,search_location),search_size)
    search_location = to_cartesian(image,new_location)
    #cropped_grey = to_greyscale(cropped)
    
    resized_grey,rel_scale = resize_image(cropped_grey,smaller_size)
    processed_grey = edge_canny(resized_grey)
    
    #processed = to_color(processed_grey)
    
    contours = find_filter_closed_contours(processed_grey)
    #draw_contours(processed,contours)
    #scaled_up_processed,rel_scale_l = resize_image(processed,400)
    
    frame_track_data = contours_to_trackData(contours)
    frame_track_data = scale_move_to_global_image(image,rel_scale[0],search_location,search_size,frame_track_data)
    oldSearchSize = search_size
    oldSearchLocation = search_location
    found = False
    if len(video_track_data)>0:
        video_track_data,velocity,found = track_contour(frame_track_data,video_track_data,5,not_found_count)
        if not found:
            not_found_count += 1
            if not_found_count > 5:
                not_found_count = 0
                video_track_data = []
                search_location = (0,0)
                search_size = 400
                velocity = (0,0)
        else:
            not_found_count = 0
        if len(video_track_data)>1:
            time_now_ms = time.time()*1000
            image_search_loc_x = video_track_data[-1].cx+velocity[0]*(time_now_ms-video_track_data[-2].timestamp)
            image_search_loc_y = video_track_data[-1].cy+velocity[1]*(time_now_ms-video_track_data[-2].timestamp)
            image_search_loc = (image_search_loc_x,image_search_loc_y)
            search_location = to_cartesian(image,image_search_loc)
            areas = np.array([p.area for p in video_track_data])
            search_size = min(max(int(np.mean(areas)*5),200),image.shape[:2][0])
    else:
        for data in frame_track_data:
            if point_distance(to_cartesian(image,(data.cx,data.cy)),(0,0)) < 150:
                video_track_data.append(data)
                break
    if found:
        draw_bounding_boxes(image,contours,rel_scale[0],color=(0,255,0),search_center=oldSearchLocation,search_size = oldSearchSize)
    else:
        draw_bounding_boxes(image,contours,rel_scale[0],color =(255,0,0),search_center=oldSearchLocation,search_size = oldSearchSize)
    draw_search_area(image,oldSearchLocation, oldSearchSize)
    #cropped,rel_scale_l = resize_image(cropped,400)
    image = rescale_image(image,0.5)
    #cv2.imwrite(f"frame_{count:04d}.png", image)  # Save frame as PNG
        #show_input(image)
    #show_input(cropped)
    #show_processed(scaled_up_processed)
    #success, image = vidcap.read()
    image = picam.capture_array()
    image = image[:frame_size[1],:frame_size[0]]
    #image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    count += 1
    g_time = count*0.033
    if count >1359:
        slow_down = True
        count = count
    if count > 1800:
        break
    #if count%30 == 0: 
    end_time = time.time()*1000
    print(f"\rLocation: {search_location} Velocity: {velocity} Frame Time: {(end_time-start_time)/count}ms",end = "")
    #start_time = time.time()*1000
    #time.sleep(0.022)
    #if slow_down:
    #    time.sleep(0.1)
picam.stop()
