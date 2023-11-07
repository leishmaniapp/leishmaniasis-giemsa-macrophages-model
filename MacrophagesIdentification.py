import time
import cv2
import cvzone
import numpy as np
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
    myColorFinder = ColorFinder(False)
    # Custom Color
    hsvVals = {'hmin': 70, 'smin': 180, 'vmin': 104, 'hmax': 255, 'smax': 255, 'vmax': 255}
    #Color Detection
    imgColor, mask = myColorFinder.update(img,hsvVals)

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
    mask2 = np.zeros((1944,1944 , 1), dtype = np.uint8)
    cores = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 160, param1=50, param2=6, minRadius=110, maxRadius=120)
    circlesTest = img.copy()
    if cores is not None:
        cores = np.uint16(np.around(cores))
        for (x,y,r ) in cores[0,:]:
            cv2.circle(circlesTest, (x,y), r,(0,0,255),3)
            cv2.circle(circlesTest, (x,y), 2,(0,0,255),3)

    cv2.imwrite('CircleTest.jpg',circlesTest)
    #Getting rid of the not rounded contours
    contours, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cores is not None:
        for (x,y,r ) in cores[0,:]:
            for i in range(len(contours)):
                cnt = contours[i]
                if 1 ==  cv2.pointPolygonTest(cnt,(x,y),False):
                    cv2.drawContours(mask2,contours,i,color=(255,255,255),thickness=cv2.FILLED)
    
    #Update found cores contours
    contours, hierarchy1 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Writing the image files
    cv2.imwrite('MaskTest.jpg',mask2)
    cv2.imwrite('CoreMask.jpg',mask)
    cv2.imwrite('CoreColored.jpg',imgColor)

    return mask2, imgColor, contours

#Cytoplasm Identification
def cytoplasmIdentification (img, coreMask, coreImgColor, cores):
    myColorFinder = ColorFinder(False)
    # Custom Color
    #hsv(285, 20%, 86%)
    hsvVals = {'hmin': 14, 'smin': 29, 'vmin': 60, 'hmax': 360, 'smax': 255, 'vmax': 255}
    #Color Detection
    imgColor, mask = myColorFinder.update(img,hsvVals)
    cv2.imwrite('CytoplasmColored.jpg',imgColor)
    diff = cv2.subtract(imgColor, coreImgColor)
    cv2.imwrite('Cytoplasm.jpg',diff)

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
    cv2.imwrite('CytoCore.jpg',diff2)

    #Blank mask
    mask2 = np.zeros((1944,1944 , 1), dtype = np.uint8)
    #Checking if the cytoplasm detected are atatch to a sutable core
    contours, hierarchy = cv2.findContours(diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Getting rid of not false macrophages identified
    for i in range(len(cores)):
        #Cores center
        cx, cy = polygonCenter(cores[i])
        for j in range(len(contours)):
            #Cytoplasm center
            x, y = polygonCenter(contours[j])
            print(distanceCalculate((x,y),(cx,cy)))
            if 450 > distanceCalculate((x,y),(cx,cy)):
                cv2.drawContours(mask2,contours,j,color=(255,255,255),thickness=cv2.FILLED)
                cv2.drawContours(mask2,cores,i,color=(255,255,255),thickness=cv2.FILLED)

    cv2.imwrite('MacropMask.jpg',mask2)

    return mask2
    
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
    cv2.imwrite('Contours.jpg',imgContours)

#Metric to measure the accuracy of the algorithm
def meanSquereError(labeled,mask):
    h, w = labeled.shape
    diff = cv2.subtract(labeled, mask) #The Difference between the two images
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    cv2.imwrite('ImageDiff.jpg',diff)
    print("Mean Squere  Error = "+str(mse))

    return mse

#This only works for masks (binary images)
def jaccardIndex(labeled, mask):
    vp = 0
    vn = 0
    fp = 0
    fn = 0
    for i in range(len(labeled)):
        for j in range(len(labeled[i])):
            if (labeled[i,j]==255 and mask[i,j]==255):
                vp+=1
            if (labeled[i,j]==0 and mask[i,j]==0):
                vn+=1
            if (labeled[i,j]==0 and mask[i,j]==255):
                fp+=1
            if (labeled[i,j]==255 and mask[i,j]==0):
                fn+=1
    print("VP = "+str(vp)+'\n'+"VN = "+str(vn)+'\n'+"FP = "+str(fp)+'\n'+"FN = "+str(fn))
    if((vp+fp+fn)==0):
        print("Jaccard Index = 0")
        return 1
    ji = vp/(vp+fp+fn)
    print("Jaccard Index = "+str(ji))
    return ji

#Dice coefficient = 2 * (intersection of sets) / (size of set 1 + size of set 2)
def SorensenDiceCoeff (labeled, mask):
    coeff = 0
    vp = 0
    vn = 0
    intersection = np.sum(mask[labeled==255])*2.0
    coeff= intersection/(np.sum(mask)+np.sum(labeled))
    
    print("Sorensen-Dice Coefficient = "+str(coeff))
    
    return coeff

#Reading the Image from the local storage
path = 'L3_M2_C13.png'

start_time = time.time()
img = cv2.imread(path,1)

#Preprocess Image - Core Identification (Part 1)
coreMask, coreImgColor , cores = coreIdentification(img)

#Preprocess Image - Cytoplasm Identification (Part2)
mask = cytoplasmIdentification(img, coreMask, coreImgColor, cores)

#Process Image
processImage(mask,img)

end_time = time.time()
execution_time = end_time - start_time
print("Segmentation Execution time:",execution_time)

dynamicThresholding()
hls()



#start_time = time.time()
#maskL=cv2.imread('Masks\\'+path,0)

#maskM=cv2.imread("CytoCore.jpg", 0)

#Metrics to measure masks
#meanSquereError(labeled=maskL,mask=maskM)
#jaccardIndex(labeled=maskL,mask=maskM)
#SorensenDiceCoeff(labeled=maskL,mask=maskM)

#end_time = time.time()
#execution_time2 = end_time - start_time
#print("Metrics Execution time:",execution_time2)