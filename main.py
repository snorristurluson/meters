import time
import sys

from camera import VideoCamera
from hotwatermeter import HotWaterMeter


def main():
    meter = HotWaterMeter()
    meter.background = "digits_threshold"

    camera = VideoCamera(meter.process_image)
    while True:
        camera.get_frame()
        time.sleep(0.8)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
