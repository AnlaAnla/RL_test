import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def imgradient(img, sobel):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel)
    return np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# 仅获取卡片斜率角度
def find_rotation_angle(im):
    im_shape = im.shape[:2]
    im_area = im_shape[0] * im_shape[1]

    # convert color to gray and
    # im_orig = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # slightly blur image to reduce noise
    im = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=2)

    # get gradient
    im_grad = imgradient(im, 3).astype(np.uint8)

    # threshold the gradient image
    _, im_grad_th = cv2.threshold(im_grad, 15, 255, cv2.THRESH_OTSU)
    im_grad_th = im_grad_th.astype(np.uint8)

    # morphological closing to take out small elements
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    im_grad_th = cv2.morphologyEx(im_grad_th, cv2.MORPH_CLOSE, kernel)
    im_grad_th_color = cv2.cvtColor(im_grad_th, cv2.COLOR_GRAY2BGR)  # debug


    # find contours
    min_shape = 0.05 * im_area  # min area a contour must have to be considered a potential card
    real_card_ratio = 0.72  # actual dimension ratio of a card (6.4 / 8.9 centimeters)

    contours, _ = cv2.findContours(im_grad_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_contour_candidates = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)  # bbx for easier drawing
        area_box = cv2.contourArea(box)

        # Check 1 for "is this contours a card ?" : is the contour big enough
        if area_box >= min_shape:

            # debug
            cv2.drawContours(im_grad_th_color, [box], -1, (0, 0, 255),
                             2)  # on peut drawer le bounding box avec drawContours

            # Check 2 : is the dimension ratio correct ?
            L1 = distance(box[0], box[1])
            L2 = distance(box[1], box[2])
            if (min(L1 / L2, L2 / L1) >= real_card_ratio - 0.05) and min(L1 / L2, L2 / L1) <= real_card_ratio + 0.05:
                # print(L1/L2)
                cv2.drawContours(im_grad_th_color, [box], -1, (255, 0, 255),
                                 2)  # on peut drawer le bounding box avec drawContours
                # draw all candidate contours
                cv2.drawContours(im_contour_candidates, [box], -1, 255,2)


    # we then find contours another time from the bbox image, to get correct external contours (watch out for inner border)
    if len(contours) >= 2:
        contours, _ = cv2.findContours(im_contour_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im_contour = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    if len(contours) == 1:
        rect = cv2.minAreaRect(contours[0])
        angle = rect[-1]
        box = cv2.boxPoints(rect).astype(int)

    # if we still have several possible contours, take the one with the ratio closest to the real ratio.
    if len(contours) >= 2:
        im_contour = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
        ratios = []
        angles = []
        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            angles.append(rect[-1])
            box = cv2.boxPoints(rect).astype(int)

            if len(contours) >= 2:
                L1 = distance(box[0], box[1])
                L2 = distance(box[1], box[2])
                if (0.67 * L2 <= L1 <= 0.77 * L2) or (
                        0.67 * L1 <= L2 <= 0.77 * L1):  # ratio officiel 0.72
                    ratios.append(min(L1 / L2, L2 / L1))
                    boxes.append(box)

        # get the closest contour
        closest_index = np.argmax(abs(ratios - 0.72))
        contours = contours[closest_index]
        angle = angles[closest_index]
        box = boxes[closest_index]

    # Define the points
    P1 = np.array(box[0])
    P2 = np.array(box[3])

    # Calculate direction vector
    dir_vec = P2 - P1

    # Vertical vector
    vertical_vec = np.array([0, 1])

    # Compute the angle using the dot product formula
    cos_theta = np.dot(dir_vec, vertical_vec) / (np.linalg.norm(dir_vec) * np.linalg.norm(vertical_vec))
    angle = np.arccos(cos_theta) * (180 / np.pi)  # Convert to degrees

    # Determine direction of tilt (left or right)
    # If the x-coordinate of P2 is greater than the x-coordinate of P1, the tilt is to the right (clockwise)
    if P2[0] > P1[0]:
        angle = -angle
    return angle


dir_path = r'C:\Code\ML\Image\angle_data\test\img'
for img_name in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_name)
    image = cv2.imread(img_path)
    rotation_angle = find_rotation_angle(image)
    print(f"{img_name}: 角度: {rotation_angle} ")

# image = cv2.imread(r"C:\Code\ML\Image\angle_data\test\img\right4 (6).jpg")
# # image = cv2.imread("slowpoke.jpg")
#
# rotation_angle = find_rotation_angle(image)
# print(f"角度: {rotation_angle} ")
