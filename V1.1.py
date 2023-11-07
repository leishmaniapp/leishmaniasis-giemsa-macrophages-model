import cv2
import numpy as np
import cvzone
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
def coreIdentification(brgImg):
    #Reding input image
    img = cv2.cvtColor(brgImg, cv2.COLOR_BGR2GRAY)
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
    #Blank Mask
    mask = np.zeros((1944,1944 , 1), dtype = np.uint8)
    if cores is not None:
        cores = np.uint16(np.around(cores))
        for (x,y,r ) in cores[0,:]:
            for i in range(len(contours)):
                cnt = contours[i]
                if 1 ==  cv2.pointPolygonTest(cnt,(x,y),False):
                    cv2.drawContours(mask,contours,i,color=(255,255,255),thickness=cv2.FILLED)
    #Update Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #GOTO: Cytoplasm Identification
    cytoplasmIdentification(mask,brgImg)

    #Cytoplasm Identification
def cytoplasmIdentification (coreMask,brgImg):
    #Writing the imge's HLS version
    cv2.imwrite('hls.jpg',cv2.cvtColor(brgImg, cv2.COLOR_RGB2HLS))
    myColorFinder = ColorFinder(False)
    # Custom Color
    hsvVals = {'hmin': 70 , 'smin': 0, 'vmin': 0, 'hmax': 150, 'smax': 200, 'vmax': 255}
    #Color Detection
    imgColor, mask = myColorFinder.update(cv2.imread('hls.jpg'),hsvVals)
    #Finding Contours in the cytoplasm mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Cleaning the mask
    minArea =7000
    for i in range(len(contours)):
        if cv2.contourArea(contours[i])<minArea:
            cv2.drawContours(mask,contours,i,color=(0,0,0),thickness=cv2.FILLED)
        else:
            cv2.drawContours(mask,contours,i,color=(255,255,255),thickness=cv2.FILLED)
    #Update Cytoplasm Contours
    cytoCnts, cytoHierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Update Core Contours
    coreCnts, coreHierarchy = cv2.findContours(coreMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Blank mask
    mask2 = np.zeros((1944,1944 , 1), dtype = np.uint8)
    #Join the Masks
    for i in range(len(cytoCnts)):
        for j in range(len(coreCnts)):
            if (distanceCalculate(polygonCenter(cytoCnts[i]),polygonCenter(coreCnts[j]))<400):
                cv2.drawContours(mask2,cytoCnts,i,color=(255,255,255),thickness=cv2.FILLED)
                cv2.drawContours(mask2,coreCnts,j,color=(255,255,255),thickness=cv2.FILLED)
    #filtering the way too big contours detected
    macpCnt, macpHierarchy = cv2.findContours(coreMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(macpCnt)):     #328123
        if(cv2.contourArea(macpCnt[i])>300000):
            cv2.drawContours(mask2,macpCnt,i,color=(0,0,0),thickness=cv2.FILLED)
        if(cv2.contourArea(macpCnt[i])<70000):
            cv2.drawContours(mask2,macpCnt,i,color=(0,0,0),thickness=cv2.FILLED)
    #TEST
    cv2.imwrite('hlsTest.jpg',imgColor)
    cv2.imwrite('CytoMask.jpg',mask2)
    #GOTO Processing Images
    processImage(mask2,brgImg)

#Processing Images
def processImage(mask, brgImg):
    print('\n'+"Macrophages Identification V3")
    #Finfing the Contours
    imgContours, conFound =  cvzone.findContours(brgImg,mask,minArea=50000)
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
    cv2.imwrite('Contours.jpg',imgContours)

#HSV COLORS
#L19_M1_C4.png
#{'hmin': 0 , 'smin': 0, 'vmin': 0, 'hmax': 115, 'smax': 70, 'vmax': 255}
#L3_M2_C12.png
#hsvVals = {'hmin': 70 , 'smin': 0, 'vmin': 0, 'hmax': 150, 'smax': 200, 'vmax': 255}

coreIdentification(cv2.imread('L3_M2_C12.png'))