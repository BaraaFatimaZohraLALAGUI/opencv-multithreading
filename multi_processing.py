
import numpy as np
import cv2 as cv
import imutils
import os
import multiprocessing
import time

def get_vcap(channel):
    ip = "10.1.67.111"
    RTSP_PORT = "554"
    USER = "admin"
    PASS = "C@meraUSTO"
    RTSP_LINK = f"rtsp://{USER}:{PASS}@{ip}:{RTSP_PORT}/cam/realmonitor?channel={channel}&subtype=0"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    return cv.VideoCapture(RTSP_LINK, cv.CAP_FFMPEG)

def read_cam(channel, frames_dict):
    # vcap = cv.VideoCapture(0, cv.CAP_DSHOW)
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

if __name__ == '__main__':
    NUM_CHANNELS = 4
    manager = multiprocessing.Manager()
    frames_dict = manager.dict()
    fgMasks_buffer = manager.dict()

    processes = []

    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=read_cam, args=(i, frames_dict))
        processes.append(process)
        process.start()

    while all(i not in frames_dict for i in range(1, NUM_CHANNELS + 1)):
        time.sleep(1)

    height, width = next(iter(frames_dict.values())).shape[:2]
    print(height, ' ', width)
    
    display_frame = np.ones((height * 2 + 20 - 80, width * 4 + 20 * 3 , 3), dtype=np.uint8) * 255

    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=bg_subtract, args=(i, frames_dict, fgMasks_buffer))
        processes.append(process)
        process.start()

    while True:
        if all(i in frames_dict and i in fgMasks_buffer for i in range(1, NUM_CHANNELS + 1)):
            try:
                display_frame[0:height, 0:width, :] = frames_dict[1]
                display_frame[height + 20:height * 2 + 20, 0:width, :] = cv.cvtColor(fgMasks_buffer[1], cv.COLOR_GRAY2BGR)

                display_frame[0:height, width + 20:width * 2 + 20, :] = frames_dict[2]
                display_frame[height + 20:height * 2 + 20, width + 20:width * 2 + 20, :] = cv.cvtColor(fgMasks_buffer[2], cv.COLOR_GRAY2BGR)

                display_frame[0:height, width * 2 + 20 + 20 : width * 2 + 20 + 20 + width, :] = frames_dict[3]  
                display_frame[height + 20:height * 2 + 20, width * 2 + 40:width * 3 + 40, :] = cv.cvtColor(fgMasks_buffer[3], cv.COLOR_GRAY2BGR)

                display_frame[0:height, width * 3 + 60:, :] = frames_dict[4]  
                display_frame[height + 20:height * 2 + 20, width * 3 + 60:width * 4 + 60, :] = cv.cvtColor(fgMasks_buffer[4], cv.COLOR_GRAY2BGR)



            except KeyError as e:
                print(f"Frame data is missing for channel: {e}")
                continue

            cv.imshow('Multi-Channel Video Feed', imutils.resize(display_frame, width=1550))

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    for process in processes:
        process.terminate()
        process.join()

    cv.destroyAllWindows()
