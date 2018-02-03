import cv2
from flask import render_template, Response, Flask

from dials import process_dials
from digits import process_digits
from main import HotWaterMeter


class VideoCamera(object):
    def __init__(self, processor):
        self.video = cv2.VideoCapture(0)
        self.processor = processor

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image = self.processor(image)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/dials')
def dials():
    meter = HotWaterMeter()

    camera = VideoCamera(meter.process_image)

    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)