import numpy as np
import cv2 as cv
import imutils
import os
import multiprocessing
import time
from yolo_api import detect
from quueuue import RetrievalQueues

NUM_CHANNELS = 4
COUNT_THRESHOLD = 100
FRAME_SWITCH_THRESHOLD = 2
DETECTION_THRESHOLD = 0.60
RECT_COLOR = (0, 0, 255)  # BGR color for detection rectangle
CLOCK_HEIGHT = 80
SPACE_COL = 20
SPACE_ROW = 20
NUM_ROWS = 3
PADDING = 20


'''
the issues:
    - multiprocessing queues are not subscriptable
    - I cannot get an element without popping it from the queue 
    - sol: rewrite the queue class by adding a get attribute method
    https://stackoverflow.com/questions/55721325/how-to-extract-multiprocessing-queue-elements-without-removing
    https://github.com/python/cpython/blob/main/Lib/multiprocessing/queues.py
    
'''

def get_vcap(channel):
    ip = "10.1.67.111"
    RTSP_PORT = "554"
    USER = "admin"
    PASS = "C@meraUSTO"
    RTSP_LINK = f"rtsp://{USER}:{PASS}@{ip}:{RTSP_PORT}/cam/realmonitor?channel={channel}&subtype=0"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    return cv.VideoCapture(RTSP_LINK, cv.CAP_FFMPEG)



def read_cam(channel, frame_queue, stop_event):
    vcap = get_vcap(channel)
    # vcap = cv.VideoCapture(0, cv.CAP_DSHOW)
    while not stop_event.is_set():
        ret, frame = vcap.read()
        if ret:
            frame_queue.put(frame)
        else:
            print(f"Channel {channel} failed to capture frame.")
            break
    vcap.release()


def bg_subtract(frames_queue, fgMasks_queue, stop_event):
    bg_subtractor = cv.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=3, backgroundRatio=0.95, noiseSigma=10)
    while not stop_event.is_set():
        if frames_queue.qsize()>0:
            frame = frames_queue[0][80:, :, :]
            fgmask = bg_subtractor.apply(frame)
            fgMasks_queue.put(fgmask)


def bound_white_space(frames_queues, fgMasks_queues, bound_fr_fg, cropped_queue, x, y, w, h, stop_event):
    prv_max_idx = -1
    while not stop_event.is_set():
        if fgMasks_queues[1].qsize() > 0 and fgMasks_queues[2].qsize() > 0 and fgMasks_queues[3].qsize() > 0 and fgMasks_queues[4].qsize() > 0:
            masks = [fgMasks_queues[i][0] for i in range(1, NUM_CHANNELS + 1)]
            ch_counts = [np.count_nonzero(masks[i]) for i in range(NUM_CHANNELS)]
            current_max_idx = np.argmax(ch_counts)

            if ch_counts[current_max_idx] > COUNT_THRESHOLD:
                if prv_max_idx == -1 or ch_counts[prv_max_idx] < COUNT_THRESHOLD or (current_max_idx != prv_max_idx and ch_counts[current_max_idx] >= FRAME_SWITCH_THRESHOLD * ch_counts[prv_max_idx]):
                    prv_max_idx = current_max_idx

                x, y, w, h = cv.boundingRect(masks[prv_max_idx])
                cropped_queue.put((x, y, w, h, frames_queues[prv_max_idx + 1][0][80:, :, :]))
                bound_fr_fg[1].put(cv.rectangle(cv.cvtColor(fgMasks_queues[prv_max_idx + 1][0], cv.COLOR_GRAY2BGR), (x, y), (x + w, y + h), (0, 255, 0), 2))

            # else:
            #     bound_fr_fg[0] = np.ones_like (frames_queues[1][0][80:, :, :], dtype=np.uint8) * 255
            #     bound_fr_fg[1] = np.ones_like (frames_queues[1][0][80:, :, :], dtype=np.uint8) * 255



def detection(cropped_queue, bound_fr_fg, x, y, w, h, stop_event):
    while not stop_event.is_set():
        if not cropped_queue.empty():
            x, y, w, h, frame = cropped_queue.get()
            people_found, frame = detect(x, y, w, h, frame, model_name='YOLOv9 Tiny', detection_threshold=DETECTION_THRESHOLD, rect_color=RECT_COLOR)
            bound_fr_fg[0].put(frame)
            print(people_found)

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    
    frames_queues = {i: RetrievalQueues(maxsize=5) for i in range(1, NUM_CHANNELS + 1)}
    fgMasks_queues = {i: RetrievalQueues(maxsize=5) for i in range(1, NUM_CHANNELS + 1)}
    
    cropped_queue = multiprocessing.Queue(maxsize=5)
    bound_fr_fg = {0: RetrievalQueues(maxsize=5), 1: RetrievalQueues(maxsize=10)}
    
    stop_event = multiprocessing.Event()
    processes = []
    x, y, w, h = 0, 0, 0, 0

    # Start camera reading processes
    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=read_cam, args=(i, frames_queues[i], stop_event))
        processes.append(process)
        process.start()

    while len(frames_queues) < NUM_CHANNELS:
        time.sleep(1)

    # Get frame dimensions from the first channel's frame
    while (frames_queues[1].qsize () == 0):
        pass
    height, width = frames_queues[1][0].shape[:2]

    display_frame = np.ones((height * NUM_ROWS + SPACE_ROW * (NUM_ROWS - 1) - CLOCK_HEIGHT * (NUM_ROWS - 1),
                             width * NUM_CHANNELS + SPACE_COL * (NUM_CHANNELS - 1), 3), dtype=np.uint8) * 255

    # Launch background subtraction processes
    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=bg_subtract, args=(frames_queues[i], fgMasks_queues[i], stop_event))
        processes.append(process)
        process.start()

    process_bound = multiprocessing.Process(target=bound_white_space, args=(frames_queues, fgMasks_queues, bound_fr_fg, cropped_queue, x, y, w, h, stop_event))
    processes.append(process_bound)
    process_bound.start()

    process_detect = multiprocessing.Process(target=detection, args=(cropped_queue, bound_fr_fg, x, y, w, h, stop_event))
    processes.append(process_detect)
    process_detect.start()

    while True:
        if all(i in frames_queues and i in fgMasks_queues for i in range(1, NUM_CHANNELS + 1)):
            try:
                display_frame[0:height, 0:width, :] = frames_queues[1].get()
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, 0:width, :] = cv.cvtColor(fgMasks_queues[1].get(), cv.COLOR_GRAY2BGR)

                display_frame[0:height, width + SPACE_COL:width * 2 + SPACE_COL, :] = frames_queues[2].get()
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, width + SPACE_COL:width * 2 + SPACE_COL, :] = cv.cvtColor(fgMasks_queues[2].get(), cv.COLOR_GRAY2BGR)

                display_frame[0:height, width * 2 + SPACE_COL * 2:width * 3 + SPACE_COL * 2, :] = frames_queues[3].get()
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, width * 2 + SPACE_COL * 2:width * 3 + SPACE_COL * 2, :] = cv.cvtColor(fgMasks_queues[3].get(), cv.COLOR_GRAY2BGR)

                display_frame[0:height, width * 3 + SPACE_COL * 3:, :] = frames_queues[4].get()
                display_frame[height + SPACE_ROW:height * 2 + SPACE_ROW - CLOCK_HEIGHT, width * 3 + SPACE_COL * 3:, :] = cv.cvtColor(fgMasks_queues[4].get(), cv.COLOR_GRAY2BGR)

                if bound_fr_fg[0].qsize() > 0 and bound_fr_fg[1].qsize() > 0:
                    print ('----------------------------------------------------- bounded frame')
                    display_frame[height * 2 + SPACE_ROW * 2 - CLOCK_HEIGHT:height * 3 + SPACE_ROW * 2 - CLOCK_HEIGHT * 2, width + SPACE_COL:width * 2 + SPACE_COL, :] = bound_fr_fg[0].get()
                    display_frame[height * 2 + SPACE_ROW * 2 - CLOCK_HEIGHT:height * 3 + SPACE_ROW * 2 - CLOCK_HEIGHT * 2, width * 2 + SPACE_COL * 2:width * 3 + SPACE_COL * 2, :] = bound_fr_fg[1].get()
            except Exception as e:
                print(f"Error displaying frame: {e}")
                continue

            cv.imshow("Frame", imutils.resize(display_frame, width=1550))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    stop_event.set()
    for process in processes:
        process.join()
    stop_event.clear()
    
    cv.destroyAllWindows()











