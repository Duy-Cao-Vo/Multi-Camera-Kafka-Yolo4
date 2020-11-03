from importlib import import_module
from flask import Flask, render_template, Response
import cv2
import time
import camera_server
import camera_yolo


app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera_stream, feed_type, topic):
    """Video streaming generator function."""
    unique_name = (feed_type, topic)

    num_frames = 0
    total_time = 0
    while True:
        time_start = time.time()

        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        num_frames += 1

        time_now = time.time()
        total_time += time_now - time_start
        fps = num_frames / total_time

        # write camera name
        cv2.putText(frame, cam_id, (int(20), int(20 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0], (255, 255, 255), 2)

        if feed_type == 'yolo':
            cv2.putText(frame, "FPS: %.2f" % fps, (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],
                        (255, 255, 255), 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # Remove this line for test camera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<feed_type>/<topic>')
def video_feed(feed_type, topic):
    """Video streaming route. Put this in the src attribute of an img tag."""
    server_list = ['localhost:9092', 'localhost:9094']
    if feed_type == 'camera':
        # camera_stream = import_module('camera_server').Camera
        camera_stream = camera_server.Camera
        return Response(gen(camera_stream=camera_stream(feed_type, topic, server_list), feed_type=feed_type, topic=topic),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    elif feed_type == 'yolo':
        # camera_stream = import_module('camera_yolo').Camera
        camera_stream = camera_yolo.Camera
        return Response(gen(camera_stream=camera_stream(feed_type, topic, server_list), feed_type=feed_type, topic=topic),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
