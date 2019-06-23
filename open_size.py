import cv2
import numpy as np
import time
import sys
from datetime import datetime

path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(path)

ratio = 0
min = 1
max = 1
contourArea = 0


def mouse_setting(event, x, y, flags, params):
    global min, max
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Min set")
        min = contourArea
    elif event == cv2.EVENT_LBUTTONUP:
        print("Max set")
        max = contourArea


def calculateContours(image, contours):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    maxArea = 0
    secondMax = 0
    maxCount = 0
    secondmaxCount = 0
    for i in contours:
        count = i
        area = cv2.contourArea(count)
        if maxArea < area:
            secondMax = maxArea
            maxArea = area
            secondmaxCount = maxCount
            maxCount = count
        elif (secondMax < area):
            secondMax = area
            secondmaxCount = count

    return [secondmaxCount, secondMax]


def thresholdContours(mouthRegion, rectArea):
    global ratio, contourArea
    imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))

    ret, thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    returnValue = calculateContours(mouthRegion, contours)
    # returnValue[0] => secondMaxCount
    # returnValue[1] => Area of the contoured region.
    secondMaxCount = returnValue[0]
    contourArea = returnValue[1]

    if contourArea<max:
        ratio = contourArea / max
    elif contourArea>max:
        ratio = 1
    else:
        ratio = 0
    if (isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
        cv2.drawContours(mouthRegion, [secondMaxCount], 0, (255, 0, 0), -1)


def yawnDetector(video_capture):
    global ratio
    ret, frame = video_capture.read()

    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        widthOneCorner = int((x + (w / 4)))
        widthOtherCorner = int(x + ((3 * w) / 4))
        heightOneCorner = int(y + (11 * h / 16))
        heightOtherCorner = int(y + h)

        cv2.rectangle(frame, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner), (0, 0, 255), 2)
        mouthRegion = frame[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]

        rectArea = (w * h) / 2

        if (len(mouthRegion) > 0):
            thresholdContours(mouthRegion, rectArea)
            # print("Prawdopodobieństwo ziewania: " + str(round(ratio * 1000, 2)) + "%")

        if (ratio > 0):
           print('Examination...')

    cv2.namedWindow('Badanie')
    cv2.setMouseCallback("Badanie", mouse_setting)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "current min : " + str(min), (10, 60), font, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "current max : " + str(max), (10, 90), font, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "current contour area : " + str(contourArea), (10, 120), font, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.putText(frame, "current ratio : " + str(round(ratio*100,2)) + "%", (10, 150), font, 1, (0, 255, 0), 2,
                cv2.LINE_AA)

    cv2.imshow('Badanie', frame)

    time.sleep(0.025)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

    return False


def main():
    yawnCamera = cv2.VideoCapture(0)

    while True:
        returnValue = (yawnDetector(yawnCamera), 'yawn')
        if returnValue[0]:
            yawnCamera.release()
            cv2.destroyWindow('yawnVideo')
            return returnValue

main()
