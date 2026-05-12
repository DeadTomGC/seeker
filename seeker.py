import cv2
import numpy as np
import time
import math

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
        
def rescale_bounding_boxes_to_image(img, contours, scale, search_center,search_size):
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = int(x/float(scale)), int(y/float(scale)), int(w/float(scale)), int(h/float(scale))
        #if search_center and search_size:
        image_search_center = to_image(img,search_center)
        x = x + (image_search_center[0]-search_size//2)
        y = y + (image_search_center[1]-search_size//2)
        bounding_boxes.append([x, y, w, h])
    return bounding_boxes

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
    def __init__(self, center,area,time_ms):
        self.area = area
        self.timestamp = time_ms
        self.cx, self.cy = center[0],center[1]

def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0, 0
    return M["m10"] / M["m00"], M["m01"] / M["m00"]

def track_contour(contours: list, past_data: list, max_data_length,missed_tracks,time_ms):
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

        dt = time_ms - past_data[-1].timestamp
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

def contours_to_trackData(contours,time_ms):
    return [trackDataPoint(get_contour_center(c),max_bounding_box_length(c),time_ms) for c in contours]

def scale_move_to_global_image(img,scale,center,size,trackData):
    frame_center = to_image(img,center)
    crop_top_left = (frame_center[0]-size//2,frame_center[1]-size//2)
    return [trackDataPoint((p.cx/scale+crop_top_left[0],p.cy/scale+crop_top_left[1]),p.area/scale,p.timestamp) for p in trackData]

def point_distance(p1, p2):
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])


class Seeker:
    """
    Tracks objects in a video stream.
    Accepts either YUV420 or BGR images.
    
    Parameters
    ----------
    frame_size : tuple
        (width, height) of the input frames.
    image_format : str
        'YUV420' or 'BGR'. Determines how greyscale is extracted from input frames.
    """

    def __init__(self, frame_size=(640, 480), image_format='YUV420'):
        self.frame_size = frame_size
        self.image_format = image_format  # 'YUV420' or 'BGR'

        # --- tracking state (mirrors original script globals) ---
        self.entity_detected = False
        self.entity_location = (0, 0)
        self.search_size = 400
        self.search_location = (0, 0)
        self.real_search_size = 400
        self.real_search_location = (0, 0)
        self.smaller_size = 256
        self.search_obj_size_ratio = 5
        self.track_history_length = 5
        self.video_track_data = []
        self.not_found_count = 0
        self.slow_down = False
        self.velocity = (0, 0)

        # --- per-frame outputs exposed for callers ---
        self.bounding_boxes = []        # rescaled bounding boxes in image coords
        self.contours = []              # raw contours from last processed crop
        self.rel_scale = (1.0, 1.0)    # scale used in last resize
        self.found = False              # whether tracking succeeded last frame
        self.resized_grey = None        # the small greyscale crop (smaller_size x smaller_size)
        self.processed_grey = None
        self.tiny_grey = None           # 40x40 version for transmission

        # internal reference image shape (set on first process call)
        self._img_shape = None

        self.count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_greyscale(self, image):
        """Convert input image to greyscale according to the configured format."""
        if self.image_format == 'YUV420':
            return yuv_to_greyscale(image)
        else:
            return to_greyscale(image)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, image, time_ms):
        """
        Process a single frame and update tracking state.

        Parameters
        ----------
        image : np.ndarray
            Raw camera frame in either YUV420 or BGR format as configured.

        Returns
        -------
        found : bool
            Whether a tracked object was found in this frame.
        """
        # Keep a reference shape for coordinate helpers
        # For YUV420 the luma plane is the top frame_size[1] rows, so we treat
        # the greyscale image as (h, w) = frame_size.  We build a dummy header
        # array just to carry the shape into the module-level helpers.

        # Extract greyscale from the raw frame
        grey_full = self._to_greyscale(image)
        # grey_full is now a 2-D array of shape (h, w)

        cropped_grey, new_location = crop_square(
            grey_full,
            to_image(grey_full, self.search_location),
            self.search_size
        )
        self.search_location = to_cartesian(grey_full, new_location)
        self.real_search_location = self.search_location
        self.real_search_size = self.search_size
        self.resized_grey, self.rel_scale = resize_image(cropped_grey, self.smaller_size)
 

        self.processed_grey = edge_canny(self.resized_grey)

        #processed = to_color(processed_grey)

        self.contours = find_filter_closed_contours(self.processed_grey)
        #draw_contours(processed, self.contours)
        #scaled_up_processed, rel_scale_l = resize_image(processed, 400)

        frame_track_data = scale_move_to_global_image(
            grey_full, self.rel_scale[0], self.search_location, self.search_size, contours_to_trackData(self.contours,time_ms)
        )

        new_found = False
        if len(self.video_track_data) > 0:
            self.video_track_data, self.velocity, new_found = track_contour(
                frame_track_data, self.video_track_data, self.track_history_length, self.not_found_count,time_ms
            )
            if not new_found:
                self.not_found_count += 1
                if self.not_found_count > 5:
                    self.not_found_count = 0
                    self.video_track_data = []
                    self.search_location = (0, 0)
                    self.search_size = 400
                    self.velocity = (0, 0)
            else:
                self.not_found_count = 0
            if len(self.video_track_data) > 1:
                image_search_loc_x = (
                    self.video_track_data[-1].cx
                    + self.velocity[0] * (time_ms - self.video_track_data[-2].timestamp)
                )
                image_search_loc_y = (
                    self.video_track_data[-1].cy
                    + self.velocity[1] * (time_ms - self.video_track_data[-2].timestamp)
                )
                image_search_loc = (image_search_loc_x, image_search_loc_y)
                self.search_location = to_cartesian(grey_full, image_search_loc)
                areas = np.array([p.area for p in self.video_track_data])
                self.search_size = min(
                    max(int(np.mean(areas) * self.search_obj_size_ratio), 200),
                    grey_full.shape[:2][0]
                )
        else:
            for data in frame_track_data:
                if point_distance(to_cartesian(grey_full, (data.cx, data.cy)), (0, 0)) < 150:
                    self.video_track_data.append(data)
                    break

        self.found = new_found

        # Compute bounding boxes in image coordinates.
        # rescale_bounding_boxes_to_image uses img.shape, so pass grey_full.
        self.bounding_boxes = rescale_bounding_boxes_to_image(
            grey_full, self.contours, self.rel_scale[0], self.real_search_location, self.real_search_size
        )

        # Small thumbnail for transmission
        tiny_grey, _ = resize_image(self.resized_grey, 40)
        self.tiny_grey = tiny_grey

        self.count += 1

        return new_found

    # ------------------------------------------------------------------
    # Debug / monitoring image helpers
    # ------------------------------------------------------------------

    def get_debug_bgr_image(self, image):
        """
        Return a BGR debug image with bounding boxes and search area overlaid.

        Parameters
        ----------
        image : np.ndarray
            The same frame that was passed to process_frame (YUV420 or BGR).

        Returns
        -------
        debug_img : np.ndarray
            BGR image at half resolution with annotations drawn on it.
        """
        # Convert to BGR for display
        if self.image_format == 'YUV420':
            bgr = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
        else:
            bgr = image #no copy

        # Draw bounding boxes
        if self.found:
            draw_bounding_boxes(
                bgr, self.contours, self.rel_scale[0],
                color=(0, 255, 0),
                search_center=self.real_search_location,
                search_size=self.real_search_size
            )
        else:
            draw_bounding_boxes(
                bgr, self.contours, self.rel_scale[0],
                color=(255, 0, 0),
                search_center=self.real_search_location,
                search_size=self.real_search_size
            )

        draw_search_area(bgr, self.real_search_location, self.real_search_size)

        # Scale down for display
        debug_img = rescale_image(bgr, 0.5)
        return debug_img

    def get_resized_grey_image(self):
        """
        Return the small greyscale crop used for edge detection (smaller_size x smaller_size).

        Returns
        -------
        np.ndarray or None
        """
        return self.resized_grey

    def get_tiny_grey_image(self):
        """
        Return the 40x40 greyscale thumbnail used for transmission.

        Returns
        -------
        np.ndarray or None
        """
        return self.tiny_grey

    def get_edge_image(self):
        """
        Return the Canny edge image of the last processed crop.

        Returns
        -------
        np.ndarray or None
        """

        return self.processed_grey

    def get_status(self):
        """
        Return a dict summarising the current tracker state.

        Returns
        -------
        dict with keys: found, search_location, search_size, velocity,
                        bounding_boxes, not_found_count, count
        """
        return {
            'found': self.found,
            'search_location': self.real_search_location,
            'search_size': self.real_search_size,
            'velocity': self.velocity,
            'bounding_boxes': self.bounding_boxes,
            'not_found_count': self.not_found_count,
            'count': self.count,
        }
