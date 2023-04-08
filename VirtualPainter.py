import cv2 as cv
import numpy as np
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
xp, yp = 0, 0
brushSize = 15
eraserSize = 100
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255, 255, 255)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1.Import Image
    success, img = cap.read()
    img = cv.flip(img, 1)

    # 2. Find HLM
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)

        # Tip of Index and Middle Finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

    # 4. If Selection Mode = Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 370 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 255, 255)
                elif 635 < x1 < 705:
                    header = overlayList[2]
                    drawColor = (150, 81, 239)
                elif 890 < x1 < 960:
                    header = overlayList[3]
                    drawColor = (0, 0, 255)
                elif 1125 < x1 < 1180:
                    header = overlayList[1]
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv.FILLED)

        # 5. If Drawing Mode = Index finger is up
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserSize)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserSize)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushSize)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushSize)

            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Setting the Header Image
    img[0:125,0:1280] = header
    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("Video", img)
    cv.waitKey(1)
