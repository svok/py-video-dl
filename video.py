from typing import Tuple

import cv2
from keras.preprocessing import image
from numpy.core.multiarray import ndarray

from model.VGG16Model import VGG16Model


def main():
    cap = cv2.VideoCapture(0)
    inference = VGG16Model()
    success: bool = True

    while success:
        # Capture frame-by-frame
        res: Tuple[bool, ndarray] = cap.read()
        success: bool = res[0]
        frame: ndarray = res[1]
        img: ndarray = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        print(inference.infern(img))

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        # cv2.imshow('frame',gray)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
