"""
leishmaniasis.giemsa.macrophages
Version 1.1

Macrophage detection in Leishmaniasis samples with Giemsa tinction
Analysis model built by _Nicolás Pérez Fonseca (nicolasperezfonseca1@gmail.com)_ in 2023.
Code cleanup and ALEF adaptation by _Angel David Talero (angelgotalero@outlook.com)_ in 2024.
"""

import logging
import numpy as np
from cvzone.ColorModule import ColorFinder
import cvzone
import cv2

# Configuration parameters
REQUIRED_RESOLUTION = 1944
"""
    Exact image resolution (in pixels) required by the model
    Both width and height MUST be this exact length in pixels
"""

ELEMENT_NAME = "macrophage"
"""
    Name of the element to diagnose
"""

# Get an already created logger
logger = logging.getLogger()


def euclideanDistanceBetween(p1, p2):
    """
    Calculate the distance between two given points in euclidean space
    """
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def polygonCenter(contour):
    """
    Calculate the center of a contour polygon
    """

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    x, y, w, h = cv2.boundingRect(approx)
    cx, cy = x + (w // 2), y + (h // 2)
    return cx, cy


def nucleusIdentification(image):
    """
    Identify the macrophages nuclei in the provided image

    Args:
        image (_type_): Raw image file

    Returns:
        _type_: mask, color and contours of the nuclei
    """

    # Create the custom color finder
    colorFinder = ColorFinder(False)

    # Custom Color
    hsvVals = {'hmin': 70, 'smin': 180, 'vmin': 104,
               'hmax': 255, 'smax': 255, 'vmax': 255}

    # Color detection
    imgColor, mask = colorFinder.update(image, hsvVals)

    # Find the contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, color=(255,
                     255, 255), thickness=cv2.FILLED)
    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask, contours, i, color=(
                0, 0, 0), thickness=cv2.FILLED)

    # Check if the contours detected are actual nuclei
    mask2 = np.zeros(
        (REQUIRED_RESOLUTION, REQUIRED_RESOLUTION, 1), dtype=np.uint8)
    nuclei = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1,
                              160, param1=50, param2=6, minRadius=110, maxRadius=120)

    circlesTest = image.copy()
    if nuclei is not None:
        nuclei = np.uint16(np.around(nuclei))
        for (x, y, r) in nuclei[0, :]:
            cv2.circle(circlesTest, (x, y), r, (0, 0, 255), 3)
            cv2.circle(circlesTest, (x, y), 2, (0, 0, 255), 3)

    # Getting rid of non rounded contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if nuclei is not None:
        for (x, y, r) in nuclei[0, :]:
            for i in range(len(contours)):
                cnt = contours[i]
                if 1 == cv2.pointPolygonTest(cnt, (x, y), False):
                    cv2.drawContours(mask2, contours, i, color=(
                        255, 255, 255), thickness=cv2.FILLED)

    # Update found cores contours
    contours, _ = cv2.findContours(
        mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return mask2, imgColor, contours


def cytoplasmIdentification(image, nucleiMask, nucleiColor, nucleiContours):
    """
    Identify the macrophages cythoplasm in the provided image

    Args:
        image (_type_): Loaded raw image
        nucleiMask (_type_): Mask returned from the 'nucleusIdentification' function
        nucleiColor (_type_): Color returned from the 'nucleusIdentification' function
        nucleiContours (_type_): Contours returned from the 'nucleusIdentification' function

    Returns:
        cv2.typing.MatLike: Cytoplasm mask
    """

    # Custom Color
    colorFinder = ColorFinder(False)

    # hsv(285, 20%, 86%)
    hsvVals = {'hmin': 14, 'smin': 29, 'vmin': 60,
               'hmax': 360, 'smax': 360, 'vmax': 360}

    # Color Detection
    imgColor, mask = colorFinder.update(image, hsvVals)

    # Substract the nuclei from the image color
    cv2.subtract(imgColor, nucleiColor)

    # Find the contours of the cytoplasm
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, color=(255,
                     255, 255), thickness=cv2.FILLED)

    minArea = 7000
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(mask, contours, i, color=(
                0, 0, 0), thickness=cv2.FILLED)

    diff = cv2.subtract(mask, nucleiMask)
    kernel = np.ones((5, 5), np.uint8)
    diff = cv2.erode(diff, kernel, iterations=1)

    contours, _ = cv2.findContours(
        diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(diff, contours, -1, color=(255,
                     255, 255), thickness=cv2.FILLED)
    minArea = 50000

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < minArea:
            cv2.drawContours(diff, contours, i, color=(
                0, 0, 0), thickness=cv2.FILLED)

    kernel = np.ones((5, 5), np.uint8)
    diff = cv2.dilate(diff, kernel, iterations=1)
    contours, _ = cv2.findContours(
        diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a blank mask
    mask = np.zeros(
        (REQUIRED_RESOLUTION, REQUIRED_RESOLUTION, 1), dtype=np.uint8)

    # Checking if the cytoplasm detected are atatch to a sutable core
    contours, _ = cv2.findContours(
        diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Getting rid of not false macrophages identified
    for i in range(len(nucleiContours)):
        # Nuclei center
        cx, cy = polygonCenter(nucleiContours[i])
        for j in range(len(contours)):

            # Cytoplasm center
            x, y = polygonCenter(contours[j])
            if 200 > euclideanDistanceBetween((x, y), (cx, cy)):
                cv2.drawContours(mask, contours, j, color=(
                    255, 255, 255), thickness=cv2.FILLED)
                cv2.drawContours(mask, nucleiContours, i, color=(
                    255, 255, 255), thickness=cv2.FILLED)

    return mask


def processContours(processedImage, rawImage):
    """
    Gather the contour data ('x' and 'y' coordinates of the center of mass and area) of the identified macrophages

    Args:
        processedImage (cv2.typing.MatLike): Image processed by `cytoplasmIdentification`
        rawImage (cv2.typing.MatLike): Original image file

    Returns:
        list[dict[str, int]]: List with each macrophage information
    """

    # Output results
    results = []

    # Find the contours the Contours
    _, contoursFound = cvzone.findContours(
        rawImage, processedImage, minArea=7000)

    if contoursFound:
        for contour in contoursFound:

            # Gather metadata
            peri = cv2.arcLength(contour['cnt'], True)
            area = contour['area']
            center = contour['center']

            # Append metadata to the results
            results.append({
                "x": center[0],
                "y": center[1],
                "area": area,
                "perimeter": peri
            })

    # Return the results
    return results


def analyze(filepath):
    """
    Start the analysis of an image

    Args:
        filepath (str): Path (can be either absolute or relative) to the image file

    Returns:
        dict[str, list[tuple[int, int]]]: A dictionary with each element found and a
        list of tuples with the coordinates of each instance
    """

    # Read the image file from path
    img = cv2.imread(filepath, 1)
    if img is None:
        raise ValueError(f"cannot open file path ({filepath})")

    logger.info(f"successfully read image ({filepath})")

    # Check if image size matches
    width, height, _ = img.shape
    if width != REQUIRED_RESOLUTION or height != REQUIRED_RESOLUTION:
        raise ValueError(
            f"image shape({width}x{height}) does not match required resolution of({REQUIRED_RESOLUTION})")

    # 1. Nuclei identification
    nucleiMask, nucleiColor, nucleiContours = nucleusIdentification(img)
    logger.info("nucleus identification successfull")

    # 2. Cytoplash identification
    cytoplasmMask = cytoplasmIdentification(
        img, nucleiMask, nucleiColor, nucleiContours)
    logger.info("cytoplasm identification successfull")

    # 3. Process image and gather contour data
    results: list = processContours(cytoplasmMask, img)

    # 4. Transform into tuples
    results = list(map(lambda it: (it["x"], it["y"]), results))

    # 5. Put the element in the dictionary
    return {ELEMENT_NAME: results}
