from imutils.video import VideoStream
import imagezmq
import cv2

# path = "rtsp://172.16.60.235:8080/h264_ulaw.sdp"  # change to your IP stream address
# cap = VideoStream(path)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')  # change to IP address and port of server thread
cam_id = 'Camera 1'  # this name will be displayed on the corresponding camera stream
cap = cv2.VideoCapture(0)
# stream = cap.start()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (736, 480),  # fx=0.6, fy=0.6,
                       interpolation=cv2.INTER_AREA)
    sender.send_image(cam_id, frame)
