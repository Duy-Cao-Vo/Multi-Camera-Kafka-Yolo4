import sys
import time
import cv2
from kafka import KafkaProducer
from imutils.video import VideoStream
import os
import re
import shutil

topic = "1"
REMOVE_LOGS = False


def publish_camera(PATH):
    """
    Publish camera video stream to specified Kafka topic.
    Kafka Server is expected to be running on the localhost. Not partitioned.
    """

    # Start up producer
    producer = KafkaProducer(bootstrap_servers='localhost:9094')

    cap = VideoStream(PATH)
    stream = cap.start()
    i = 0
    t1 = time.time()
    try:
        while True:
            frame = stream.read()
            frame = cv2.resize(frame, (736, 480),  # fx=0.6, fy=0.6,
                               interpolation=cv2.INTER_AREA)

            ret, buffer = cv2.imencode('.jpg', frame)
            i += 1
            if i % 1000 == 0:
                print("SEND", i, time.time() - t1)
                t1 = time.time()
            if i % 10 in [0, 1]:
                continue
            producer.send(topic, buffer.tobytes())

            # Choppier stream, reduced load on processor
            # time.sleep(0.2)

    except:
        print("\nExiting.")
        sys.exit(1)

def publish_video(PATH):
    """
    Publish camera video stream to specified Kafka topic.
    Kafka Server is expected to be running on the localhost. Not partitioned.
    """

    # Start up producer
    producer = KafkaProducer(bootstrap_servers='localhost:9094')

    video_capture = cv2.VideoCapture(PATH)
    i = 0
    t1 = time.time()
    try:
        while True:
            oke, frame = video_capture.read()
            if not oke:
                break
            frame = cv2.resize(frame, (736, 480),  # fx=0.6, fy=0.6,
                               interpolation=cv2.INTER_AREA)

            ret, buffer = cv2.imencode('.jpg', frame)
            i += 1
            if i % 1000 == 0:
                print("SEND", i, time.time() - t1)
                t1 = time.time()
            if i % 10 in [0, 1]:
                continue
            producer.send(topic, buffer.tobytes())

            # Choppier stream, reduced load on processor
            # time.sleep(0.2)

    except:
        print("\nExiting.")


def publish_webcam():
    """
    Publish camera video stream to specified Kafka topic.
    Kafka Server is expected to be running on the localhost. Not partitioned.
    """

    # Start up producer
    producer = KafkaProducer(bootstrap_servers='localhost:9094')

    cap = cv2.VideoCapture(0)
    i = 0
    t1 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (736, 480),  # fx=0.6, fy=0.6,
                               interpolation=cv2.INTER_AREA)

            ret, buffer = cv2.imencode('.jpg', frame)
            i += 1
            if i % 1000 == 0:
                print("SEND", i, time.time() - t1)
                t1 = time.time()
            if i % 10 in [0, 1]:
                continue
            producer.send(topic, buffer.tobytes())

            # Choppier stream, reduced load on processor
            # time.sleep(0.2)
    except:
        print("\nExiting.")


if __name__ == '__main__':
    """
    Producer will publish to Kafka Server a video file given as a system arg. 
    Otherwise it will default by streaming webcam feed.
    """
    # path = "rtsp://192.168.0.8:8080/h264_ulaw.sdp"
    # path = "rtsp://admin:ICUIFC@172.16.60.80:554/Streaming/Channels/101/"
    # publish_camera(path)

    file_path = 'D:\\video/hlc_sala_test.mp4'
    publish_video(file_path)
    # publish_webcam()
