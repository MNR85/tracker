# USAGE
# python multi_object_tracking.py --video videos/soccer_01.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our list of queues -- both input queue and output queue
# for *every* object that we will be tracking
inputQueues = []
outputQueues = [] 

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

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

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(1, 1920, 1080)
# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=0).start()
	vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"], cv2.CAP_V4L)

frameCount=0
print (1111)
# loop over frames from the video stream
while True:
	frameCount = frameCount+1
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	print (2222)
	# check to see if we have reached the end of the stream
	if frame is None:
		print (3333)
		if frameCount>30:
			break
		continue
	print (444)
	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)
	frameCount = frameCount+1

	# if our list of queues is empty then we know we have yet to
	# create our first object tracker
	if frameCount%5 == 0:
		# initialize OpenCV's special multi-object tracker
		trackers = cv2.MultiTracker_create()
		# grab the frame dimensions and convert the frame to a blob
		(h, w) = frame.shape[:2]
		t1=time.time()
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
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
			if confidence > args["confidence"]:
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
				tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
				trackers.add(tracker, frame, box)								

				# grab the corresponding class label for the detection
				# and draw the bounding box
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# otherwise, we've already performed detection so let's track
	# multiple objects
	else:
		# grab the updated bounding box coordinates (if any) for each
		# object that is being tracked
		t1=time.time()
		(success, boxes) = trackers.update(frame)
		print ('track time: ',str((time.time()-t1)*1000000))

		# loop over the bounding boxes and draw then on the frame
		for box in boxes:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the output frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()