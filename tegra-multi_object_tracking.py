# --------------------------------------------------------
# Camera sample code for Tegra X2/X1
#
# This program could capture and display video from
# IP CAM, USB webcam, or the Tegra onboard camera.
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------


import sys
import argparse
import subprocess

import cv2
import imutils
import time
import numpy as np

WINDOW_NAME = 'CameraDemo'


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    parser.add_argument("-t", "--tracker", type=str, default="kcf",help="OpenCV object tracker type", dest='a_tracker')
    parser.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file", dest='a_proto')
    parser.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model", dest='a_model')
    parser.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')


def read_cam(cap,net,args):
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()

    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN
    frameCount=0
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read()  # grab the next image frame from camera
        # if our list of queues is empty then we know we have yet to
        # create our first object tracker
        if frameCount%5 == 0:
            # initialize OpenCV's special multi-object tracker
            trackers = cv2.MultiTracker_create()
            # grab the frame dimensions and convert the frame to a blob
            (h, w) = img.shape[:2]
            t1=time.time()
            blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)
            print ('dnn time: ',str((time.time()-t1)*1000000))

            # pass the blob through the network and obtain the detections
            # and predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > 0.2: #args["confidence"]:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    bb = (startX, startY, endX, endY)
                    box = (startX, startY, endX-startX, endY - startY)

                    # create a new object tracker for the bounding box and add it
                    # to our multi-object trackerQ
                    tracker = OPENCV_OBJECT_TRACKERS[args.a_tracker]()
                    trackers.add(tracker, img, box)								

                    # grab the corresponding class label for the detection
                    # and draw the bounding box
                    cv2.rectangle(img, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                    cv2.putText(img, label, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # otherwise, we've already performed detection so let's track
        # multiple objects
        else:
            # grab the updated bounding box coordinates (if any) for each
            # object that is being tracked
            t1=time.time()
            (success, boxes) = trackers.update(img)
            print ('track time: ',str((time.time()-t1)*1000000))

            # loop over the bounding boxes and draw then on the frame
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(10)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'):  # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    
    # load our serialized model from disk
    print("[INFO] loading model...")
    print(args.a_proto)
    net = cv2.dnn.readNetFromCaffe(args.a_proto, args.a_model)

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else:  # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)
    read_cam(cap,net,args)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
