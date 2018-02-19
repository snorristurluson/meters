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
        sys.stdout.flush()
        time.sleep(1)


if __name__ == "__main__":
    main()
