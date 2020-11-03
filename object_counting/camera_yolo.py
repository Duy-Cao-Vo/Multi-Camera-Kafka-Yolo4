# from __future__ import division, print_function, absolute_import
#
# import cv2
# from base_camera import BaseCamera
#
# import warnings
# import numpy as np
# from PIL import Image
# from yolo import YOLO
# from deep_sort import preprocessing
# from deep_sort.detection import Detection
# from deep_sort.detection_yolo import Detection_YOLO
# from importlib import import_module
# from collections import Counter
# import datetime
from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from base_camera import BaseCamera

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from collections import deque
from math import sqrt, acos, tan
from imutils.video import FPS
import datetime

warnings.filterwarnings('ignore')


class fps_callback(FPS):
    def __init__(self):
        FPS.__init__(self)

    def update_time(self):
        # increment the total number of frames examined during the
        # start and end intervals
        return (datetime.datetime.now() - self._start).total_seconds()


class Camera(BaseCamera):
    def __init__(self, feed_type, device, port_list):
        super(Camera, self).__init__(feed_type, device, port_list)

    @staticmethod
    def yolo_frames(unique_name):
        device = unique_name[1]

        # Definition of the parameters
        max_cosine_distance = 0.7
        nn_budget = None
        nms_max_overlap = 3

        # Deep SORT
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        yolo = YOLO()

        show_detections = True  # show object box blue when detect
        writeVideo_flag = False  # record video ouput

        defaultSkipFrames = 2  # skipped frames between detections

        # set up collection of door
        H1 = 210
        W1 = 370
        H2 = 235
        W2 = 470
        H = None
        W = None

        R = 80  # min R is 56

        def solve_quadratic_equation(a, b, c):
            """ax2 + bx + c = 0"""
            delta = b ** 2 - 4 * a * c
            if delta < 0:
                print("Phương trình vô nghiệm!")
            elif delta == 0:
                return -b / (2 * a)
            else:
                print("Phương trình có 2 nghiệm phân biệt!")
                if float((-b - sqrt(delta)) / (2 * a)) > float((-b + sqrt(delta)) / (2 * a)):
                    return float((-b - sqrt(delta)) / (2 * a))
                else:
                    return float((-b + sqrt(delta)) / (2 * a))

        def setup_door(H1, W1, H2, W2, R):
            # bước 1 tìm trung điểm của W1, H1 W2, H2
            I1 = (W1 + W2) / 2
            I2 = (H1 + H2) / 2

            # tìm vecto AB
            u1 = W2 - W1
            u2 = H2 - H1

            # AB chính là vecto pháp tuyến của d
            # ta có phương trình trung tuyến của AB
            # y = -(u1 / u2)* x - c/u2
            c = -u1 * I1 - u2 * I2  # tìm c

            # bước 2 tìm tâm O của đường tròn
            al = c / u2 + I2
            # tính D: khoảng cách I và O
            fi = acos(sqrt((I1 - W1) ** 2 + (I2 - H1) ** 2) / R)
            D = sqrt((I1 - W1) ** 2 + (I2 - H1) ** 2) * tan(fi)

            O1 = solve_quadratic_equation((1 + u1 ** 2 / u2 ** 2), 2 * (-I1 + u1 / u2 * al), al ** 2 - D ** 2 + I1 ** 2)
            O2 = -u1 / u2 * O1 - c / u2
            # phương trình 2 nghiệm chỉ chọn nghiệm phía trên

            # Bước 3 tìm các điểm trên đường tròn
            door_dict = dict()
            for w in range(W1, W2):
                h = O2 + sqrt(R ** 2 - (w - O1) ** 2)
                door_dict[w] = round(h)
            return door_dict

        door_dict = setup_door(H1, W1, H2, W2, R)

        totalFrames = 0
        totalIn = 0

        # create a empty list of centroid to count traffic
        pts = [deque(maxlen=30) for _ in range(9999)]


        fps_imutils = fps_callback().start()

        get_feed_from = ('camera', device)

        while True:
            cam_id, frame = BaseCamera.get_frame(get_feed_from)

            # Resize frame
            # frame = cv2.resize(frame, (736, 480))
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]


            # Draw a door boundary
            for w in range(W1, W2):
                cv2.circle(frame, (w, door_dict[w]), 1, (0, 255, 255), -1)
            cv2.circle(frame, (W1, H1), 4, (0, 0, 255), -1)
            cv2.circle(frame, (W2, H2), 4, (0, 0, 255), -1)
            cv2.circle(frame, (204, 201), 4, (0, 0, 255), -1)

            if True: # totalFrames % defaultSkipFrames == 0:
                # t2 = time.time()
                boxes, confidence, classes = yolo.detect_image(image)  # average time: 1.2s
                # print(time.time() - t2)

                features = encoder(frame, boxes)
                detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                              zip(boxes, confidence, classes, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.cls for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

            # else:
            #     # Call the tracker
            #     tracker.predict()
            #     tracker.update(detections)

            for det in detections:
                bbox = det.to_tlbr()
                if show_detections and len(classes) > 0:
                    det_cls = det.cls
                    score = "%.2f" % (det.confidence * 100) + "%"
                    cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3]) - 10), 0,
                                1e-3 * frame.shape[0], (0, 255, 0), 1)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 40:
                    continue
                bbox = track.to_tlbr()

                # if not_count_staff(frame, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])):
                #     # adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
                #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 1)
                #     cv2.putText(frame, "STAFF", (int(bbox[0]), int(bbox[1]) - 10), 0,
                #                 1e-3 * frame.shape[0], (0, 0, 255), 1)
                #     continue

                # adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255),
                              1)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

                x = [c[0] for c in pts[track.track_id]]
                y = [c[1] for c in pts[track.track_id]]
                under_door = True

                centroid_x = int(((bbox[0]) + (bbox[2])) / 2)
                centroid_y = int(((bbox[1]) + (bbox[3])) / 2)

                # checker: Count person through store
                if not track.Counted and centroid_x in range(W1, W2):
                    # if all centroid of user have detect in last 30 frame in door range(W1, W2)
                    # if user is found in store so under_door is fail --> do not check through door
                    if all(u[0] in range(W1, W2) for u in pts[track.track_id]):
                        if all(u[1] < door_dict[u[0]] for u in pts[track.track_id]):
                            under_door = False
                    '''
                    check condition
                    1. person must go up: entroid_y < np.mean (y)
                    2. person must pass door circle: door_dict[centroid_x] > centroid_y
                    3. person must move around Horizontal at least 20 px: np.max (x) - np.min (x) > 20
                    4. person must have at least 1 centroid under the door
                    '''
                    if centroid_y < np.mean(y) and door_dict[centroid_x] > centroid_y and np.max(x) - np.min(x) > 20 \
                            and under_door:
                        totalIn += 1
                        track.Counted = True
                        print(track.track_id, track.Counted)

                cv2.circle(frame, (centroid_x, centroid_y), 4, (0, 255, 0), -1)
                pts[track.track_id].append((centroid_x, centroid_y))

            info = [
                ("Time", "{:.4f}".format(fps_imutils.update_time())),
                ("In", totalIn)
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (W - 150, ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            yield cam_id, frame
