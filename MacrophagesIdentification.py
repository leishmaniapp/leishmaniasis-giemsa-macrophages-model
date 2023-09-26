import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

#Pre Process Method 1
def preProcessing(img):
    myColorFinder = ColorFinder(False)
    # Custom Color
    hsvVals = {'hmin': 70, 'smin': 180, 'vmin': 104, 'hmax': 255, 'smax': 255, 'vmax': 255}
    #Color Detection
    imgColor, mask = myColorFinder.update(img,hsvVals)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask,contours, -1, color=(255,255,255),thickness=cv2.FILLED)

    imgPre = cv2.GaussianBlur(imgColor,(5,5),3)
    kernel = np.ones((8,8),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel, iterations=1)
    imgPre = cv2.Canny(imgPre,110,200,apertureSize = 3)
    imgPre = cv2.dilate(imgPre,kernel, iterations=2)
    imgPre = cv2.morphologyEx(imgPre,cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 6000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask,contours,i,color=(0,0,0),thickness=cv2.FILLED)
    cv2.imwrite('Pre Processed.jpg',imgPre)

    return imgPre

#Pre Process Method 2
def preProcessing_(img):
    myColorFinder = ColorFinder(False)
    # Custom Color
    hsvVals = {'hmin': 70, 'smin': 180, 'vmin': 104, 'hmax': 255, 'smax': 255, 'vmax': 255}
    #Color Detection
    imgColor, mask = myColorFinder.update(img,hsvVals)
    #Pre Process
    imgPre_ = cv2.GaussianBlur(imgColor,(5,5),3)
    imgPre_ = cv2.Canny(imgPre_,110,200,apertureSize = 3)
    kernel = np.ones((9,9),np.uint8)
    imgPre_ = cv2.dilate(imgPre_,kernel,iterations=1)
    imgPre_ = cv2.morphologyEx(imgPre_, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask,contours,i,color=(0,0,0),thickness=cv2.FILLED)
    cv2.imwrite('Mask.jpg',mask)
    cv2.imwrite('Colored.jpg',imgColor)
    cv2.imwrite('Pre Processed_.jpg',imgPre_)

    return imgPre_

#Cytoplasm Identification
def cytoplasmIdentification (img):
    myColorFinder = ColorFinder(False)
    # Custom Color
    #hsv(264, 23%, 88%)
    hsvVals = {'hmin': 100, 'smin': 23, 'vmin': 88, 'hmax': 360, 'smax': 360, 'vmax': 360}
    #Color Detection
    imgColor, mask = myColorFinder.update(img,hsvVals)
    cv2.imwrite('CytoplasmColored.jpg',imgColor)
    imgColor2 = cv2.imread('Colored.jpg',1)
    diff = cv2.subtract(imgColor, imgColor2)
    cv2.imwrite('Cytoplasm.jpg',diff)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask,contours, -1, color=(255,255,255),thickness=cv2.FILLED)
    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask,contours,i,color=(0,0,0),thickness=cv2.FILLED)
    cv2.imwrite('CytoMask.jpg',mask)
    coreMask = cv2.imread ('Mask.jpg',0)
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
    cv2.imwrite('MacropMask.jpg',diff2)

    return diff2


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

#Processing Images Method 2
def processImage_(imgPre, img):
    print('\n'+"Method 2")
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
    cv2.imwrite('Contours_.jpg',imgContours)

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

#Preprocessing the Image (method 1)
imgPre = preProcessing(img)

#Preprocessing the Image (method 2)
imgPre_ = preProcessing_(img)

#Cytoplasm Identification Test
mask = cytoplasmIdentification(img)

#Processing Images Pre. 1
processImage(mask,img)

#Processing Images Pre. 2
processImage_(mask,img)

#Metrics to measure masks
#mask_old = cv2.imread('Mask_old.jpg',0)
#mask = cv2.imread('Mask.jpg',0)
#meanSquereError(labeled=mask,mask=mask)
#jaccardIndex(labeled=mask,mask=mask)
#SorensenDiceCoeff(labeled=mask,mask=mask)