


import cv2
import numpy as np
import time
import math
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


def resize_image(img, size):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    scale = size / w, size / h
    return resized, scale

def to_cartesian(img, coord):
    h, w = img.shape[:2]
    return (int(coord[0] - w // 2), int(h // 2 - coord[1]))

def to_image(img, coord):
    h, w = img.shape[:2]
    return (coord[0] + w // 2, h // 2 - coord[1])

def dog_detect(img, fg_blur, bg_blur, homogeneity_threshold=3.0):
    if cv2.Laplacian(img, cv2.CV_64F).std() < homogeneity_threshold:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    if fg_blur > 0:
        fg = cv2.GaussianBlur(img, (fg_blur, fg_blur), 0)
    else:
        fg = img
    bg = cv2.GaussianBlur(img, (bg_blur, bg_blur), 0)
    dog = cv2.absdiff(fg, bg)
    _, binary = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def dog_detect_old(img, fg_blur, bg_blur, scale=2.0):

    if fg_blur > 0:
        fg = cv2.GaussianBlur(img, (0, 0), fg_blur)
    else:
        fg = img
    bg = cv2.GaussianBlur(img, (0, 0), bg_blur)
    diff = cv2.absdiff(fg, bg)
    #normed = cv2.normalize(diff,None,0,255,cv2.NORM_MINMAX)
    mult = cv2.multiply(diff,img/(255.0/scale),dtype=cv2.CV_8U)
    _, suppress = cv2.threshold(mult, 10, 255, cv2.THRESH_TOZERO)
    _, edges = cv2.threshold(suppress, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    
    return edges

def edgedetect(img, fg_blur, bg_blur, scale=2.0):

    if fg_blur > 0:
        fg = cv2.GaussianBlur(img, (0, 0), fg_blur)
    else:
        fg = img
    bg = cv2.GaussianBlur(img, (0, 0), bg_blur)
    diff = cv2.absdiff(fg, bg)
    #normed = cv2.normalize(diff,None,0,255,cv2.NORM_MINMAX)
    mult = cv2.multiply(diff,img/(255.0/scale),dtype=cv2.CV_8U)
    _, suppress = cv2.threshold(mult, 10, 255, cv2.THRESH_TOZERO)
    _, edges = cv2.threshold(suppress, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    
    return edges

def auto_canny(image, sigma=0.05):
    # Compute median of single channel pixel intensities
    v = np.mean(image)
    
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    return cv2.Canny(image, lower, upper)

def auto_edge_canny(img):
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return auto_canny(blur)

def edge_canny(img, low=50, high=200):
    #blur = cv2.GaussianBlur(img, (0, 0), 1)
    return cv2.Canny(img, low, high)

def to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def normalize_polarity(img, tolerance=0.0):
    count = cv2.countNonZero(img)
    ratio = count / img.size
    if abs(ratio - 0.5) < tolerance:
        return np.zeros_like(img)
    if ratio > 0.5:
        return cv2.bitwise_not(img)
    return img

def find_filter_closed_contours(edge_img,kernel_size = 5,min_solidity = 0.4):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    #closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        #perimeter = cv2.arcLength(c,True)
        if hull_area > 0 and area / hull_area >= min_solidity:
        #if perimeter>0 and area/perimeter >= min_solidity:
            good_contours.append(c)
    return good_contours

def find_filter_closed_contours_2(edge_img,kernel_size = 7,min_solidity = 1):
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

def filter_contours(img, min_pixels=1, max_pixels=2000):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if min_pixels <= cv2.contourArea(c) <= max_pixels]

def draw_bounding_boxes(img, contours, scale, color=(0, 255, 0), thickness=2):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

def draw_contours(img, contours, color=(0, 255, 0), thickness=2):
    cv2.drawContours(img, contours, -1, color, thickness)

class trackDataPoint:
    def __init__(self, center,area):
        self.area = area
        self.timestamp = g_time * 1000
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

        dt = g_time * 1000 - past_data[-1].timestamp
        exp_x = past_data[-1].cx + vx * dt
        exp_y = past_data[-1].cy + vy * dt
        exp_area = np.mean(areas)
    else:
        exp_x = past_data[-1].cx
        exp_y = past_data[-1].cy
        exp_area = past_data[-1].area

    # larger margin when little data
    margin_scale = max(1.0, 3.0 / n)
    margin_scale = margin_scale + (0.5*missed_tracks)
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

video_name = 'short_video.mp4'

vidcap = cv2.VideoCapture(video_name)
success, image = vidcap.read()
count = 0

entity_detected = False
entity_location = (0,0)
search_size = 400
search_location = (0,0)
smaller_size = 256
video_track_data = []
not_found_count = 0
if success:
    print(f"Video Opened: {video_name}")
else:
    print("Error, Exiting")
while success:
    #TODO process image
    
    cropped,new_location = crop_square(image,to_image(image,search_location),search_size)
    cropped_grey = to_greyscale(cropped)
    resized_grey,rel_scale = resize_image(cropped_grey,smaller_size)
    processed_grey = edge_canny(resized_grey)
    #processed_grey = normalize_polarity(processed_grey)
    
    processed = to_color(processed_grey)
    
    contours = find_filter_closed_contours_2(processed_grey)
    draw_contours(processed,contours)
    scaled_up_processed,rel_scale_l = resize_image(processed,400)
    draw_bounding_boxes(cropped,contours,rel_scale[0])
    frame_track_data = contours_to_trackData(contours)
    frame_track_data = scale_move_to_global_image(image,rel_scale[0],search_location,search_size,frame_track_data)
    

    if len(video_track_data)>0:
        video_track_data,velocity,found = track_contour(frame_track_data,video_track_data,5,not_found_count)
        if not found:
            not_found_count += 1
            if not_found_count > 30:
                not_found_count = 0
                video_track_data = []
                search_location = (0,0)
                search_size = 400
                velocity = (0,0)
        else:
            not_found_count = 0
        if len(video_track_data)>1:
            time_now_ms = g_time*1000
            image_search_loc_x = video_track_data[-1].cx+velocity[0]*(time_now_ms-video_track_data[-2].timestamp)
            image_search_loc_y = video_track_data[-1].cy+velocity[1]*(time_now_ms-video_track_data[-2].timestamp)
            image_search_loc = (image_search_loc_x,image_search_loc_y)
            search_location = to_cartesian(image,image_search_loc)
            areas = np.array([p.area for p in video_track_data])
            search_size = max(int(np.mean(areas)*5),200)
    else:
        for data in frame_track_data:
            if point_distance(to_cartesian(image,(data.cx,data.cy)),(0,0)) < 150:
                video_track_data.append(data)
                break
    cropped,rel_scale_l = resize_image(cropped,400)
    show_input(cropped)
    show_processed(scaled_up_processed)
    success, image = vidcap.read()
    count += 1
    g_time = count*0.033
    if count >1159:
        count = count
    if count > 1800:
        break
    time.sleep(0.022)