"""Run this script from the root of the project to test the LandmarkExtractor on a webcam feed."""

import numpy as np

from gesture_transformer.datasets.landmark_extractor import LandmarkExtractor
import cv2 as cv


def main() -> None:
    """Run a demo of the LandmarkExtractor on a webcam feed."""
    extractor = LandmarkExtractor(max_num_hands=1, flip_horizontal=True, model_asset_path="src/gesture_transformer/models/hand_landmarker.task")

    cap = cv.VideoCapture('data/raw/test.mp4')
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extractor.extract(frame)
        for i in range(0, len(features), 3):
            x, y, z = features[i:i + 3]
            if x == 0 and y == 0 and z == 0:
                continue
            cv.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

        cv.imshow("Webcam Feed", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
