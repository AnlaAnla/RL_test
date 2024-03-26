import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path_debug = './debug'

def imgradient(img, sobel):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel)
    return np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def rotateImage(image, angle, nocrop=False):
    dt = image.dtype

    # handling case of color and grayscale images
    if len(image.shape) == 2:
        color = False
    elif len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    else:
        raise Exception("Incorrect image shape for rotateImage")
    h = image.shape[0]
    w = image.shape[1]

    # compute the rotation matrix based on the desired angle
    if nocrop:
        diag = np.sqrt(w ** 2 + h ** 2).astype(np.uint64)
        if color:
            img2 = np.zeros((diag, diag, 3))
        else:
            img2 = np.zeros((diag, diag))
        img2[int(diag / 2 - h / 2):int(diag / 2 + h / 2), int(diag / 2 - w / 2):int(diag / 2 + w / 2)] = image
        image = img2.copy()
        resulting_shape = (diag, diag)
        rot_mat = cv2.getRotationMatrix2D((int(diag / 2), int(diag / 2)), angle * 180 / np.pi, 1.0)
    else:
        resulting_shape = (w, h)
        image_center = tuple(np.array([w, h]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle * 180 / np.pi, 1.0)

    # apply rotation matrix
    if color:
        result0 = cv2.warpAffine(image[:, :, 0], rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
        result1 = cv2.warpAffine(image[:, :, 1], rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
        result2 = cv2.warpAffine(image[:, :, 2], rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
        result = np.zeros(image.shape)
        result[:, :, 0] = result0
        result[:, :, 1] = result1
        result[:, :, 2] = result2
    else:
        result = cv2.warpAffine(image, rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
    return result.astype(dt)


# 获取卡片
def find_card_in_img(im, debug=False):
    im_shape = im.shape[:2]
    im_area = im_shape[0] * im_shape[1]
    im_orig = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=2)
    im_grad = imgradient(im, 3).astype(np.uint8)
    _, im_grad_th = cv2.threshold(im_grad, 15, 255, cv2.THRESH_OTSU)
    im_grad_th = im_grad_th.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    im_grad_th = cv2.morphologyEx(im_grad_th, cv2.MORPH_CLOSE, kernel)
    min_shape = 0.05 * im_area
    real_card_ratio = 0.72

    if debug:
        path_debug = './debug'
        cv2.imwrite(os.path.join(path_debug, "im_grad_th.jpg"), im_grad_th)

    contours, _ = cv2.findContours(im_grad_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_contour_candidates = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    card_contours = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)
        area_box = cv2.contourArea(box)

        if area_box >= min_shape:
            L1 = distance(box[0], box[1])
            L2 = distance(box[1], box[2])
            ratio = min(L1 / L2, L2 / L1)
            if real_card_ratio - 0.05 <= ratio <= real_card_ratio + 0.05:
                card_contours.append(contour)

    if debug and card_contours:
        im_grad_th_color = cv2.cvtColor(im_grad_th, cv2.COLOR_GRAY2BGR)
        for contour in card_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(im_grad_th_color, [box], -1, (255, 0, 255), 2)
        cv2.imwrite(os.path.join(path_debug, "contours_debug.png"), im_grad_th_color)

    if len(card_contours) >= 1:
        ratios = []
        boxes = []
        angles = []
        for contour in card_contours:
            rect = cv2.minAreaRect(contour)
            angle = rect[-1]
            box = cv2.boxPoints(rect).astype(int)
            L1 = distance(box[0], box[1])
            L2 = distance(box[1], box[2])
            ratio = min(L1 / L2, L2 / L1)
            if real_card_ratio - 0.05 <= ratio <= real_card_ratio + 0.05:
                ratios.append(ratio)
                boxes.append(box)
                angles.append(angle)

        real_card_ratio_list = [real_card_ratio] * len(ratios)
        closest_index = np.argmin(np.abs(np.array(ratios) - np.array(real_card_ratio_list)))
        contour = card_contours[closest_index]
        angle = angles[closest_index]
        box = boxes[closest_index]
    elif len(card_contours) == 1:
        contour = card_contours[0]
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        box = cv2.boxPoints(rect).astype(int)
    else:
        return None, None

    im_contour = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    cv2.drawContours(im_contour, [box], -1, 255, 2)

    if debug:
        cv2.imwrite(os.path.join(path_debug, "contour.png"), im_contour)

    cv2.fillPoly(im_contour, pts=[contour], color=255)
    im_card = cv2.bitwise_and(im_orig, im_orig, mask=im_contour)

    if debug:
        cv2.imwrite(os.path.join(path_debug, "detected_card.png"), im_card)

    rotated_im = rotateImage(im_card, +(angle + 90) / 180 * np.pi, nocrop=True)
    rotated_mask = rotateImage(im_contour, +(angle + 90) / 180 * np.pi, nocrop=True)
    contour, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contour[0])
    im_card_isol = rotated_im[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

    if debug:
        cv2.imwrite(os.path.join(path_debug, "rotated_card.png"), rotated_im)
        cv2.imwrite(os.path.join(path_debug, "isolated_card.png"), im_card_isol)

    return im_card_isol, box



# dir_path = r"C:\Code\ML\Image\angle_data\test\img"
# for filename in os.listdir(dir_path):
#     img_path = os.path.join(dir_path, filename)
#     image = cv2.imread(img_path)
#
#     isolated_card, card_box = find_card_in_img(image,False)
#     if isolated_card is not None:
#         corrected_card = cv2.rotate(isolated_card, cv2.ROTATE_180)
#     else:
#         corrected_card = image
#
#     print(filename)
#     save_path = os.path.join(r'C:\Code\ML\Image\angle_data\correct', filename)
#     cv2.imwrite(save_path, corrected_card)


image = cv2.imread(r"C:\Code\ML\Image\angle_data\test\img\2024_03_22___03_41_50.jpg")
isolated_card, card_box = find_card_in_img(image,True)
corrected_card = cv2.rotate(isolated_card, cv2.ROTATE_180)
cv2.imwrite('path_to_save_image.jpg', corrected_card)
# plt.imshow(cv2.cvtColor(corrected_card, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # Hide axes
# plt.show()