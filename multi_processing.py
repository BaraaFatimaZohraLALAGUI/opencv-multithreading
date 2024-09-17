import numpy as np
import cv2 as cv
import imutils
import os
import multiprocessing



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
            frame = frames_dict[channel][80:, :, :]  # Process the region of interest
            fgmask = bg_substractor.apply(frame)
            fgMasks_buffer[channel] = fgmask




if __name__ == '__main__':
    NUM_CHANNELS = 4
    manager = multiprocessing.Manager()
    frames_dict = manager.dict()
    fgMasks_buffer = manager.dict()

    processes = []

    # Start processes for each channel (for reading video)
    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=read_cam, args=(i, frames_dict))
        processes.append(process)
        process.start()

  
    # Start background subtraction processes
    for i in range(1, NUM_CHANNELS + 1):
        process = multiprocessing.Process(target=bg_subtract, args=(i, frames_dict, fgMasks_buffer))
        processes.append(process)
        process.start()

    while True:
        # Ensure all channels have frames and background masks before display
        if all(i in frames_dict and i in fgMasks_buffer for i in range(1, NUM_CHANNELS + 1)):
            frames = [frames_dict[i] for i in range(1, NUM_CHANNELS + 1)]
            bgs = [cv.cvtColor(fgMasks_buffer[i], cv.COLOR_GRAY2BGR) for i in range(1, NUM_CHANNELS + 1)]

            display_fr = np.hstack(frames)  
            display_bg = np.hstack(bgs)  

            display = np.vstack((display_fr, display_bg))  # Stack original and mask frames vertically
            cv.imshow('Multi-Channel Video Feed', imutils.resize(display, width=1550))

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    # Properly terminate the processes
    for process in processes:
        process.terminate()
        process.join()

    cv.destroyAllWindows()
