from base_camera import BaseCamera
import time
import cv2
import numpy as np
from kafka import KafkaConsumer


class Camera(BaseCamera):

    def __init__(self, feed_type, topic, server):
        super(Camera, self).__init__(feed_type, topic, server)

    @classmethod
    def server_frames(cls, topic, server):
        num_frames = 0
        total_time = 0
        print(" ----------------------------------------------------------------------------------------------- ")
        print("DEBUB TOPIC", topic, server)
        consumer = KafkaConsumer(
            str(topic),
            bootstrap_servers=server)
        while True:  # main loop
            time_start = time.time()
            for msg in consumer:
                frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), cv2.IMREAD_COLOR)
                yield topic, frame
                # num_frames += 1
                #
                # time_now = time.time()
                # total_time += time_now - time_start
                # fps = num_frames / total_time

                # uncomment below to see FPS of camera stream
                # cv2.putText(frame, "FPS: %.2f" % fps, (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],(255, 255, 255), 2)
