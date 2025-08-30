import numpy as np
import cv2


class Pupil(object):
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        kernel = np.bilateralFilter((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        _, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)

        return new_frame

    def detect_iris(self, eye_frame):
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contours.sort(key=cv2.contourArea, reverse=True)
            
            try:
                moments = cv2.moments(contours[0])
                if moments['m00'] != 0:
                    self.x = int(moments['m10'] / moments['m00'])
                    self.y = int(moments['m01'] / moments['m00'])
            except (IndexError, ZeroDivisionError):
                pass