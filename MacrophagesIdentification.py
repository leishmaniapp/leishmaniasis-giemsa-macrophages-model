import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

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
    mask2 = np.zeros((1944,2592 , 1), dtype = np.uint8)
    #just for testing the funtion and its parameters
    testImage = cv2.imread('MacropMask.jpg',0)
    cores = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 160, param1=50, param2=5, minRadius=110, maxRadius=120)
    circlesTest = img.copy()
    cores = np.uint16(np.around(cores))
    for (x,y,r ) in cores[0,:]:
        cv2.circle(circlesTest, (x,y), r,(0,0,255),3)
        cv2.circle(circlesTest, (x,y), 2,(0,0,255),3)

    cv2.imwrite('CircleTest.jpg',circlesTest)
    #Getting rid of the not rounded contours
    contours, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for (x,y,r ) in cores[0,:]:
        for i in range(len(contours)):
            cnt = contours[i]
            if 1 ==  cv2.pointPolygonTest(cnt,(x,y),False):
                cv2.drawContours(mask2,contours,i,color=(255,255,255),thickness=cv2.FILLED)
    
    #Update found cores
    cores = cv2.HoughCircles(mask2, cv2.HOUGH_GRADIENT, 1, 160, param1=50, param2=5, minRadius=110, maxRadius=120)
    circlesTest = img.copy()
    cores = np.uint16(np.around(cores))

    #Writing the image files
    cv2.imwrite('MaskTest.jpg',mask2)
    cv2.imwrite('CoreMask.jpg',mask)
    cv2.imwrite('CoreColored.jpg',imgColor)

    return mask2, imgColor, cores

#Cytoplasm Identification

def cytoplasmIdentification (img, coreMask, coreImgColor, cores):
    myColorFinder = ColorFinder(False)
    # Custom Color
    #hsv(264, 23%, 88%)
    hsvVals = {'hmin': 100, 'smin': 23, 'vmin': 88, 'hmax': 360, 'smax': 360, 'vmax': 360}
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
    cv2.imwrite('CytoCore.jpg',diff2)

    contours, hierarchy = cv2.findContours(diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(diff2,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 50000

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(diff2,contours,i,color=(0,0,0),thickness=cv2.FILLED)

    #Blank mask
    mask2 = np.zeros((1944,2592 , 1), dtype = np.uint8)
    #Checking if the cytoplasm detected are atatch to a sutable core
    contours, hierarchy = cv2.findContours(diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Getting rid of not false macrophages identified
    for (x,y,r ) in cores[0,:]:
        for i in range(len(contours)):
            cnt = contours[i]
            if 1 ==  cv2.pointPolygonTest(cnt,(x,y),False):
                cv2.drawContours(mask2,contours,i,color=(255,255,255),thickness=cv2.FILLED)

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
        print("Jaccard Index = "+str(ji))
        return 1
    ji = vp/(vp+fp+fn)
    print("Jaccard Index = "+str(ji))
    return ji

#Dice coefficient = 2 * (intersection of sets) / (size of set 1 + size of set 2)
def SorensenDiceCoeff (labeled, mask):
    coeff = 0
    vp = 0
    vn = 0
    for i in range(len(labeled)):
        for j in range(len(labeled[i])):
            if (labeled[i,j]==255 and mask[i,j]==255):
                vp+=1
            if (labeled[i,j]==0 and mask[i,j]==0):
                vn+=1
    coeff = 2*(vn+vp)/2*len(labeled)*len(labeled[1])
    print("Sorensen-Dice Coefficient = "+str(coeff))
    return coeff

#Reading the Image from the local storage
img = cv2.imread('L24_M1_C1.png',1)

#Preprocess Image - Core Identification (Part 1)
coreMask, coreImgColor , cores = coreIdentification(img)

#Preprocess Image - Cytoplasm Identification (Part2)
mask = cytoplasmIdentification(img, coreMask, coreImgColor, cores)

#Process Image
processImage(mask,img)

#Metrics to measure masks
#meanSquereError(labeled=mask,mask=CoreMask)
#jaccardIndex(labeled=mask,mask=CoreMask)111
#SorensenDiceCoeff(labeled=mask,mask=CoreMask)