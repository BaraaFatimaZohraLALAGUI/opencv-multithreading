
import numpy as np
import cv2 as cv
import imutils
import os
import multiprocessing
import time
from itertools import repeat
from yolo_api import detect


def get_vcap(channel):
    ip = "10.1.67.111"
    RTSP_PORT = "554"
    USER = "admin"
    PASS = "C@meraUSTO"
    RTSP_LINK = f"rtsp://{USER}:{PASS}@{ip}:{RTSP_PORT}/cam/realmonitor?channel={channel}&subtype=0"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    return cv.VideoCapture(RTSP_LINK, cv.CAP_FFMPEG)

def read_cam(channel, frames_dict):
    vcap = get_vcap(channel)
    while True:
        ret, frame = vcap.read()
        if ret:
            frames_dict[channel] = frame
        else:
            print(f"Channel {channel} failed to capture frame.")
            break

def bg_subtract(channel, frames_dict, fgMasks_buffer):
    bg_substractor = cv.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=3, backgroundRatio=0.95, noiseSigma=10)
    while True:
        if channel in frames_dict:
            frame = frames_dict[channel][80:, :, :] 
            fgmask = bg_substractor.apply(frame)
            fgMasks_buffer[channel] = fgmask

def bound_white_space(frames_dict, fgMasks_buffer, bound_fr_fg, bounding_queue, COUNT_THRESHOLD, FRAME_SWITCH_THRESHOLD):
    prv_max_idx = -1
    while True:
        if len(fgMasks_buffer) > 0:
            masks = [fgMasks_buffer[i] for i in range(1, 5)]
            ch_counts = [np.count_nonzero(masks[i]) for i in range(4)]
            current_max_idx = np.argmax(ch_counts)

            if ch_counts[current_max_idx] > COUNT_THRESHOLD:
                if prv_max_idx == -1 or ch_counts[prv_max_idx] < COUNT_THRESHOLD or (current_max_idx != prv_max_idx and ch_counts[current_max_idx] >= FRAME_SWITCH_THRESHOLD * ch_counts[prv_max_idx]):
                    prv_max_idx = current_max_idx

                x, y, w, h = cv.boundingRect(masks[prv_max_idx])
                bounded_rect = frames_dict[prv_max_idx + 1][y:y + h, x:x + w, :]
                bounding_queue.put(bounded_rect)
                
                bound_fr_fg[0] = frames_dict[prv_max_idx + 1]            
                bound_fr_fg[1] = cv.rectangle(cv.cvtColor(fgMasks_buffer[prv_max_idx + 1], cv.COLOR_GRAY2BGR), (x, y), (x + w, y + h), (0, 255, 0), 2)
            # else:
            #     bound_fr_fg[0] = None          
            #     bound_fr_fg[1] = None
             

def detection(bounding_queue, DETECTION_THRESHOLD, RECT_COLOR):
    while True:
        if not bounding_queue.empty ():
            bounded_rect = bounding_queue.get ()
            detect(bounded_rect, model_name='YOLOv9 Tiny', detection_threshold=DETECTION_THRESHOLD, rect_color=RECT_COLOR)


if __name__ == '__main__':
    NUM_CHANNELS = 4
    COUNT_THRESHOLD = 100
    FRAME_SWITCH_THRESHOLD = 2
    DETECTION_THRESHOLD = 0.60
    RECT_COLOR = (255, 0, 0)
    CLOCK_HEIGHT = 80
    SPACE_COL = 20
    SPACE_ROW = 20
    NUM_ROWS = 3

    manager = multiprocessing.Manager()
    frames_dict = manager.dict()
    fgMasks_buffer = manager.dict()
    bounding_queue = manager.Queue()
    bound_fr_fg = manager.dict(zip(range(2), repeat(None)))

    processes = []

    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=read_cam, args=(i, frames_dict))
        processes.append(process)
        process.start()

    while all(i not in frames_dict for i in range(1, NUM_CHANNELS + 1)):
        time.sleep(1)

    height, width = next(iter(frames_dict.values())).shape[:2]    
    display_frame = np.ones((height * NUM_ROWS + SPACE_ROW * (NUM_ROWS - 1) - CLOCK_HEIGHT * (NUM_ROWS - 1), 
                             width * NUM_CHANNELS + SPACE_COL * (NUM_CHANNELS - 1), 3), dtype=np.uint8) * 255
    SPACE2 = np.ones((height - CLOCK_HEIGHT, width, 3), dtype=np.uint8) * 255
    SPACE = np.ones((height, width, 3), dtype=np.uint8) * 255


    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=bg_subtract, args=(i, frames_dict, fgMasks_buffer))
        processes.append(process)
        process.start()

    process_bound = multiprocessing.Process(target=bound_white_space, args=(frames_dict, fgMasks_buffer, bound_fr_fg, bounding_queue, COUNT_THRESHOLD, FRAME_SWITCH_THRESHOLD))
    processes.append(process_bound)
    process_bound.start()

    process_detect = multiprocessing.Process(target=detection, args=(bounding_queue, DETECTION_THRESHOLD, RECT_COLOR))
    processes.append(process_detect)
    process_detect.start()


    while True:
        if all(i in frames_dict and i in fgMasks_buffer for i in range(1, NUM_CHANNELS + 1)):
            try:
                display_frame[0:height, 0:width, :] = frames_dict[1]
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, 0:width, :] = cv.cvtColor(fgMasks_buffer[1], cv.COLOR_GRAY2BGR)

                display_frame[0:height, width + SPACE_COL:width * 2 + SPACE_COL, :] = frames_dict[2]
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, width + SPACE_COL:width * 2 + SPACE_COL, :] = cv.cvtColor(fgMasks_buffer[2], cv.COLOR_GRAY2BGR)

                display_frame[0:height, width * 2 + SPACE_COL * 2:width * 3 + SPACE_COL * 2, :] = frames_dict[3]
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, width * 2 + SPACE_COL * 2:width * 3 + SPACE_COL * 2, :] = cv.cvtColor(fgMasks_buffer[3], cv.COLOR_GRAY2BGR)

                display_frame[0:height, width * 3 + SPACE_COL * 3:, :] = frames_dict[4]
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, width * 3 + SPACE_COL * 3:, :] = cv.cvtColor(fgMasks_buffer[4], cv.COLOR_GRAY2BGR)

                display_frame[height * 2 + SPACE_ROW * 2 - CLOCK_HEIGHT:height * 3 + SPACE_ROW * 2 - CLOCK_HEIGHT * 2, width + SPACE_COL:width * 2 + SPACE_COL, :] = SPACE2 if bound_fr_fg[0] is None else bound_fr_fg[0][80:, :, :] 
                display_frame[height * 2 + SPACE_ROW * 2 - CLOCK_HEIGHT:height * 3 + SPACE_ROW * 2 - CLOCK_HEIGHT * 2, width * 2 + SPACE_COL * 2:width * 3 + SPACE_COL * 2, :] = SPACE2 if bound_fr_fg[1] is None else bound_fr_fg[1] 


            except KeyError as e:
                print(f"Frame data is missing for channel: {e}")
                continue

            cv.imshow('Multi-Channel Video Feed', imutils.resize(display_frame, width=1550))

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    # Terminate all processes
    for process in processes:
        process.terminate()
        process.join()

    cv.destroyAllWindows()

