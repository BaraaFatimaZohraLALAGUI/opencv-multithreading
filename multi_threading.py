import cv2 as cv
import numpy as np
import os
import threading
from collections import deque
import imutils
from yolo_api import detect

def get_vcap(channel):
    ip = "10.1.67.111"
    RTSP_PORT = "554"
    USER = "admin"
    PASS = "C@meraUSTO"
    RTSP_LINK = f"rtsp://{USER}:{PASS}@{ip}:{RTSP_PORT}/cam/realmonitor?channel={channel}&subtype=0"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    return cv.VideoCapture(RTSP_LINK, cv.CAP_FFMPEG)

NUM_CHANNELS = 4
COUNT_THRESHOLD = 100
FRAME_SWITCH_THRESHOLD = 1.5
DETECTION_THRESHOLD = 0.60
RECT_COLOR = (255, 0, 0)

vcaps = [get_vcap(channel=i+1) for i in range(NUM_CHANNELS)]
bg_substractors = [cv.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=3, backgroundRatio=0.95, noiseSigma=10) for _ in range(NUM_CHANNELS)]
prv_max_idx = -1

SPACE_COL = np.ones((int(vcaps[0].get(cv.CAP_PROP_FRAME_HEIGHT)), 20, 3), dtype=np.uint8) * 255
SPACE_COL2 = np.ones((int(vcaps[0].get(cv.CAP_PROP_FRAME_HEIGHT)) - 80, 20, 3), dtype=np.uint8) * 255
SPACE_ROW = np.ones((20, int(vcaps[0].get(cv.CAP_PROP_FRAME_WIDTH)) * NUM_CHANNELS + 20 * 4, 3), dtype=np.uint8) * 255
SPACE = np.ones((int(vcaps[0].get(cv.CAP_PROP_FRAME_HEIGHT)), int(vcaps[0].get(cv.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8) * 255
SPACE2 = np.ones((int(vcaps[0].get(cv.CAP_PROP_FRAME_HEIGHT)) - 80, int(vcaps[0].get(cv.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8) * 255

frames_buffer = [deque(maxlen=5), deque(maxlen=5), deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)]
fgMasks_buffer = [deque(maxlen=5), deque(maxlen=5), deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)]
# max_buffer = [deque(maxlen=5), deque(maxlen=5)]
bounding_queue = deque(maxlen=5)

# Frames buffer locks
fr_bf_read_lock = threading.Lock()
fr_bf_modify_lock = threading.Lock()

# FgMasks buffer locks 
fg_bf_read_lock = threading.Lock()
fg_bf_modify_lock = threading.Lock()

# bounding queue locks 
bound_modify_lock = threading.Lock()


def read_cam(channel):
    while True:
        ret, frame = vcaps[channel].read()
        if ret:
            with fr_bf_modify_lock:
                frames_buffer[channel].append(frame)


def bg_subtract(channel):
    while True:
        if len(frames_buffer[channel]) > 0:
            fgmask = bg_substractors[channel].apply(frames_buffer[channel][0][80:, :, :])
            with fg_bf_modify_lock:
                fgMasks_buffer[channel].append(fgmask)


def bound_white_space():
    global bounding_queue
    global prv_max_idx
    while True:
            if (len(fgMasks_buffer[0]) > 0 and len(fgMasks_buffer[1]) > 0 and len(fgMasks_buffer[2]) > 0 and len(fgMasks_buffer[3]) > 0):
                masks = [fgMasks_buffer[i][0] for i in range(NUM_CHANNELS)]
                ch_counts = [np.count_nonzero(masks[i]) for i in range(NUM_CHANNELS)]
                current_max_idx = np.argmax(ch_counts)

                if ch_counts[current_max_idx] > COUNT_THRESHOLD:
                    if prv_max_idx == -1 or (current_max_idx != prv_max_idx and ch_counts[current_max_idx] >= FRAME_SWITCH_THRESHOLD * ch_counts[prv_max_idx]):
                        prv_max_idx = current_max_idx

                    x, y, w, h = cv.boundingRect(masks[prv_max_idx])
                    bounded_rect = frames_buffer[prv_max_idx][0][y:y+h, x:x+w]
                    with bound_modify_lock:
                        bounding_queue.append(bounded_rect)
                    with fr_bf_modify_lock:
                        frames_buffer[4].append(frames_buffer[prv_max_idx][0])
                    with fg_bf_modify_lock:
                        cv.rectangle(cv.cvtColor(masks[prv_max_idx], cv.COLOR_GRAY2BGR),(x,y),(x+w,y+h),(0,255,0),2)
                        fgMasks_buffer[4].append(masks[prv_max_idx])       


def detection():
    while True:
        with bound_modify_lock:
            if len(bounding_queue) > 0:
                bounded_rect = bounding_queue.popleft()
                detect(bounded_rect, model_name='YOLOv9 Tiny', detection_threshold=DETECTION_THRESHOLD, rect_color=RECT_COLOR)

try:

    print ('Starting threads')

    for channel in range(NUM_CHANNELS):
        vcap_thread = threading.Thread(target=read_cam, args=(channel,))
        vcap_thread.start()

        bg_thread = threading.Thread(target=bg_subtract, args=(channel,))
        bg_thread.start()

    bound_thread = threading.Thread(target=bound_white_space)
    bound_thread.start()

    detect_thread = threading.Thread(target=detection)
    detect_thread.start()


    while True:


        all_frames_ready = any(len(frames_buffer[i]) > 0 for i in range(NUM_CHANNELS))
        all_fgMasks_ready = any(len(fgMasks_buffer[i]) > 0 for i in range(NUM_CHANNELS))

        for fb in frames_buffer:
            print (f'--------------------------- {len(fb)}', end=' ')
        print ()
        for fgb in fgMasks_buffer:
            print (f'>>>>>>>>>>>>>>>>>>>>>>>>>>> {len(fgb)}', end=' ')
        
        
        print()

        if not (all_frames_ready and all_fgMasks_ready):
            continue

        with fr_bf_modify_lock:
            combined_frame = np.hstack([np.hstack(( SPACE_COL, frames_buffer[i].popleft())) if len(frames_buffer[i]) != 0  else np.hstack(( SPACE_COL, SPACE)) for i in range(NUM_CHANNELS)])
        with fg_bf_modify_lock:
            combined_bg = np.hstack([np.hstack(( SPACE_COL2, cv.cvtColor(fgMasks_buffer[i].popleft(), cv.COLOR_GRAY2BGR))) if len(fgMasks_buffer[i]) != 0 else np.hstack(( SPACE_COL2, SPACE2)) for i in range(NUM_CHANNELS)])
        
        bounded_frame = np.hstack(( SPACE_COL2, SPACE2,  SPACE_COL2, frames_buffer[4].popleft()[80:, :, :] if len(frames_buffer[4]) != 0  else SPACE2, SPACE_COL2, cv.putText(fgMasks_buffer[4].popleft(), f'Ch {prv_max_idx + 1}', fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 255, 255), bottomLeftOrigin=False, org=(int(frames_buffer[0][0].get(cv.CAP_PROP_FRAME_WIDTH)) - 200, 80), thickness=2, lineType=cv.LINE_AA) if len(fgMasks_buffer[4]) != 0  else SPACE2, SPACE_COL2, SPACE2))
        big_frame = np.vstack(( SPACE_ROW, combined_frame, SPACE_ROW, combined_bg, SPACE_ROW, bounded_frame))
        # big_frame = np.vstack(( SPACE_ROW, combined_frame, SPACE_ROW, combined_bg, SPACE_ROW))

        cv.imshow('Multi-Channel Video Feed', imutils.resize(big_frame, width=1550))

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
        

finally:
    for vcap in vcaps:
        vcap.release()
    cv.destroyAllWindows()
