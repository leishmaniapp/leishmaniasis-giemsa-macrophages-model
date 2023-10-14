import time
import cv2
import cvzone
import numpy as np
import snake_opencv as cv
from cvzone.ColorModule import ColorFinder

#Calculate the distance between two given points
def distanceCalculate(p1, p2):
    #p1 and p2 in format (x1,y1) and (x2,y2) tuples
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

#Calculate de Center of a contour polygon
def polygonCenter (contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    cx, cy = x + (w // 2), y + (h // 2)
    return cx, cy

#Pre Process Method
def coreIdentification(img):

    mask = dynamicThresholding(img)
    #Pre Process
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask,contours,i,color=(0,0,0),thickness=cv2.FILLED)

    #Checking if the contoures detected are actual cores
    mask2 = np.zeros((1944,2592 , 1), dtype = np.uint8)
    cores = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 160, param1=50, param2=6, minRadius=110, maxRadius=120)
    circlesTest = img.copy()
    cores = np.uint16(np.around(cores))
    for (x,y,r ) in cores[0,:]:
        cv2.circle(circlesTest, (x,y), r,(0,0,255),3)
        cv2.circle(circlesTest, (x,y), 2,(0,0,255),3)

    #Getting rid of the not rounded contours
    contours, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for (x,y,r ) in cores[0,:]:
        for i in range(len(contours)):
            cnt = contours[i]
            if 1 ==  cv2.pointPolygonTest(cnt,(x,y),False):
                cv2.drawContours(mask2,contours,i,color=(255,255,255),thickness=cv2.FILLED)
    
    #Update found cores contours
    contours, hierarchy1 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return mask2, contours

#Cytoplasm Identification
def cytoplasmIdentification (img, coreMask, cores):

    #CoreColors
    myColorFinder = ColorFinder(False)
    # Custom Color
    hsvVals = {'hmin': 70, 'smin': 180, 'vmin': 104, 'hmax': 255, 'smax': 255, 'vmax': 255}
    #Color Detection
    coreImgColor, mask = myColorFinder.update(img,hsvVals)

    myColorFinder = ColorFinder(False)
    # Custom Color
    #hsv(285, 20%, 86%)
    hsvVals = {'hmin': 14, 'smin': 29, 'vmin': 60, 'hmax': 360, 'smax': 360, 'vmax': 360}
    #Color Detection
    imgColor, mask = myColorFinder.update(img,hsvVals)
    diff = cv2.subtract(imgColor, coreImgColor)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask,contours,i,color=(0,0,0),thickness=cv2.FILLED)
    diff2 = cv2.subtract(mask, coreMask)
    kernel = np.ones((5,5),np.uint8)
    diff2 = cv2.erode(diff2,kernel,iterations=1)

    contours, hierarchy = cv2.findContours(diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(diff2,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 50000

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(diff2,contours,i,color=(0,0,0),thickness=cv2.FILLED)

    kernel = np.ones((5,5),np.uint8)
    diff2 = cv2.dilate(diff2,kernel,iterations=1)
    contours, hierarchy = cv2.findContours(diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Blank mask
    mask2 = np.zeros((1944,2592 , 1), dtype = np.uint8)
    #Checking if the cytoplasm detected are atatch to a sutable core
    contours, hierarchy = cv2.findContours(diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Getting rid of not false macrophages identified
    for i in range(len(cores)):
        #Cores center
        cx, cy = polygonCenter(cores[i])
        for j in range(len(contours)):
            #Cytoplasm center
            x, y = polygonCenter(contours[j])
            if 200 > distanceCalculate((x,y),(cx,cy)):
                cv2.drawContours(mask2,contours,j,color=(255,255,255),thickness=cv2.FILLED)
                cv2.drawContours(mask2,cores,i,color=(255,255,255),thickness=cv2.FILLED)

    return mask2

#Dynamic thresholding
def dynamicThresholding (img):
    #Reding input image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Apply threshold
    ret1,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Invert mask
    th1 = cv2.bitwise_not(th1)
    #Find contours
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Draw contours
    cv2.drawContours(th1,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    #Filter by area
    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(th1,contours,i,color=(0,0,0),thickness=cv2.FILLED)
    #Update Contours
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Hough Transform to check roundiness
    cores = cv2.HoughCircles(th1, cv2.HOUGH_GRADIENT, 1, 160, param1=50, param2=6, minRadius=110, maxRadius=120)
    cores = np.uint16(np.around(cores))
    #Blank Mask
    mask = np.zeros((1944,2592 , 1), dtype = np.uint8)
    for (x,y,r ) in cores[0,:]:
        for i in range(len(contours)):
            cnt = contours[i]
            if 1 ==  cv2.pointPolygonTest(cnt,(x,y),False):
                cv2.drawContours(mask,contours,i,color=(255,255,255),thickness=cv2.FILLED)
    #Update Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return mask

#Processing Images Method 1
def processImage(imgPre, img):
    print('\n'+"Method 1")
    #Finfing the Contours
    imgContours, conFound =  cvzone.findContours(img,imgPre,minArea=7000)
    #Defining what will be a macophag for us
    macrophages = 0
    if conFound:
        for contour in conFound:
            macrophages+=1
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)
            area = contour['area']
            center = contour['center']
            print ("area: "+str(area)+'\n'+"center: "+str(center))
    print("total detected: "+str(macrophages))

#Reading the Image from the local storage
start_time = time.time()
#img = cv2.imread('L3_M2_C13.png',1)
img = cv2.imread('L19_M1_C2.png',1)

#Preprocess Image - Core Identification (Part 1)
coreMask, cores = coreIdentification(img)

#Preprocess Image - Cytoplasm Identification (Part2)
mask = cytoplasmIdentification(img, coreMask, cores)

#Process Image
processImage(mask,img)

end_time = time.time()
execution_time = end_time - start_time
print("Segmentation Execution time:",execution_time)