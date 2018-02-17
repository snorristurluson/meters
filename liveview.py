from flask import render_template, Response, Flask

from camera import VideoCamera
from hotwatermeter import HotWaterMeter

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dials')
def dials():
    return render_template('dials.html')

@app.route('/digits')
def digits():
    return render_template('digits.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/dials_feed')
def dials_feed():
    meter = HotWaterMeter()
    meter.background = "dials_threshold"

    camera = VideoCamera(meter.process_image)

    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/digits_feed')
def digits_feed():
    meter = HotWaterMeter()
    meter.background = "digits_threshold"

    camera = VideoCamera(meter.process_image)

    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)