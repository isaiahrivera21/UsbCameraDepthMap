## Goal is to use openCV library to create a depth map
## Sources: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Set up external camera
    cam = cv2.VideoCapture(2)

    # Two params to tune (put these in yaml file)
    stereo = cv2.StereoBM.create(numDisparities=96, blockSize=5)

    # Display video feed and depth map
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        height, width, _ = frame.shape
        camL = frame[:, :width // 2, :]
        camR = frame[:, width // 2:, :]

        camL = cv2.cvtColor(camL, cv2.COLOR_BGR2GRAY)
        camR = cv2.cvtColor(camR, cv2.COLOR_BGR2GRAY)


        disparity = stereo.compute(camL,camR)



        # Display left and right frames
        cv2.imshow('Left Lens', camL)
        cv2.imshow('Right Lens', camR)
        cv2.imshow("USB Camera", frame)
        cv2.imshow("Depth Map", disparity)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
