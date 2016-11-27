# coding: utf-8
import cv2
import numpy as np
import math
import LineIterator


# private function
def __distance__(p, q):
    """
    euler distance
    :param p:
    :param q:
    :return:
    """
    return int(math.sqrt(pow((p[0] - q[0]), 2) + pow((p[1] - q[1]), 2)))


def __two_nearest_line__(b1, b2):
    """
    get two nearest lines between two boxes
    :param b1:
    :param b2:
    :return:
    """
    distances = []
    for p in b1:
        for q in b2:
            distances.append([__distance__(p, q), (p, q)])
    distances = sorted(distances, key=lambda d: d[0])
    return distances[0], distances[1]


# image utils
def show(img, win_name='qr code'):
    """
    show qr code image in cv window
    :param img:
    :param win_name:
    :return:
    """
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def read_image(img_path, need_convert=False):
    """
    read image
    :param img_path:
    :param need_convert:
    :return: origin image, gray image
    """
    qr_origin = cv2.imread('./test_pictures/' + img_path)
    if need_convert:
        qr_origin = cv2.cvtColor(qr_origin, cv2.COLOR_BGR2RGB)
    qr_gray = cv2.cvtColor(qr_origin, cv2.COLOR_BGR2GRAY)
    return qr_origin, qr_gray


def draw_separate_position_patterns(img, found, contours):
    """
    draw every single found position pattern separately
    :param img:
    :param found:
    :param contours:
    :return:
    """
    for i in found:
        qr_dc = img.copy()
        cv2.drawContours(qr_dc, contours, i, (0, 255, 0), 2)
        show(qr_dc)


def draw_all_position_patterns(img, found, contours):
    """
    draw all position patterns
    :param img:
    :param found:
    :param contours:
    :return:
    """
    draw_img = img.copy()
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.cv.BoxPoints(rect))
        cv2.drawContours(draw_img, [box], 0, (255, 0, 0), 2)
    show(draw_img)


def draw_box(img, box):
    """
    draw box on image
    :param img:
    :param box:
    :return:
    """
    draw_img = img.copy()
    cv2.polylines(draw_img, np.int32([box]), True, (255, 0, 0), 4)
    show(draw_img)


def draw_lines(img, boxes):
    """
    draw two nearest lines between boxes
    :param img:
    :param boxes:
    :return:
    """
    draw_img = img.copy()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            d1, d2 = __two_nearest_line__(boxes[i], boxes[j])
            cv2.line(draw_img, d1[1][0], d1[1][1], (0, 255, 0), 2)
            cv2.line(draw_img, d2[1][0], d2[1][1], (0, 255, 0), 2)
    show(draw_img)


def draw_timing_pattern(img, timing_patterns):
    """
    draw found timing patterns
    :param img:
    :param timing_patterns:
    :return:
    """
    draw_img = img.copy()
    for timing_pattern in timing_patterns:
        cv2.line(draw_img, timing_pattern[0], timing_pattern[1], (0, 255, 0), 2)
    show(draw_img)


# get function
def get_binary_image(img):
    th, qr_bi = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return th, qr_bi


def get_edges(img):
    """
    get edges from image using Canny
    :param img:
    :return: edges
    """
    edges = cv2.Canny(img, 100, 200)
    return edges


def get_contours(img):
    """
    get contours from image
    :param img:
    :return: contours and inner hierarchy
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy[0]


def get_position_patterns(contours, hierarchy):
    """
    get qr code position pattern
    :param contours:
    :param hierarchy:
    :return: found position pattern index
    """
    found = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c += 1
        if c >= 5:
            found.append(i)
    return found


def get_contours_points(found, contours):
    """
    get contour's all inner points
    :param found:
    :param contours:
    :return: contour's all inner points
    """
    contours_points = []
    for i in found:
        c = contours[i]
        for sublist in c:
            for p in sublist:
                contours_points.append(p)
    return contours_points


def get_area_box(contours_points):
    """
    get area rectangle box corner
    :param contours_points:
    :return: box corner
    """
    rect = cv2.minAreaRect(np.array(contours_points))
    box = cv2.cv.BoxPoints(rect)
    box = np.array(box)
    return map(tuple, box)


def get_boxes(found, contours):
    """
    get all boxes list in the image
    :param found:
    :param contours:
    :return:
    """
    boxes = []
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = np.int0(cv2.cv.BoxPoints(rect))
        box = map(tuple, box)
        boxes.append(box)
    return boxes


# detection function
def is_timing_pattern(line, threshold=5):
    """
    judge if line pattern is a qr code timing pattern
    :param line:
    :param threshold:
    :return:
    """
    while line[0] != 0:
        line = line[1:]
        if not len(line):
            return False
    while line[-1] != 0:
        line = line[:-1]
        if not len(line):
            return False
    c = []
    count = 1
    l = line[0]
    for p in line[1:]:
        if p == l:
            count += 1
        else:
            c.append(count)
            count = 1
        l = p
    c.append(count)
    if len(c) < 5:
        return False
    return np.var(c) < threshold


def get_valid_boxes_index(boxes, qr_bi):
    """
    get valid position pattern due to timing pattern
    :param boxes:
    :param qr_bi:
    :return: box index and timing patterns
    """
    timing_patterns = []
    boxes_dict = {}
    boxes_index = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            d1, d2 = __two_nearest_line__(boxes[i], boxes[j])
            for d in [d1[1], d2[1]]:
                line_pixels = LineIterator.createLineIterator(d[0], d[1], qr_bi)
                if is_timing_pattern(line_pixels[:, 2]):
                    timing_patterns.append((d[0], d[1]))
                    boxes_index.append((i, j))
    for i in range(len(boxes_index)):
        for j in range(i+1, len(boxes_index)):
            common = set(boxes_index[i]).intersection(set(boxes_index[j]))
            if common:
                boxes_dict[common.pop()] = set(boxes_index[i] + boxes_index[j])
    return boxes_dict, timing_patterns


def get_qr_code_boxes(boxes_dict, contours, found):
    """
    get qr code outer box, maybe multiple boxes
    :param boxes_dict:
    :param contours:
    :param found:
    :return:
    """
    boxes = dict()
    for id, boxes_index in boxes_dict.items():
        contours_points = []
        while len(boxes_index) > 0:
            c = contours[found[boxes_index.pop()]]
            for sublist in c:
                for p in sublist:
                    contours_points.append(p)
        rect = cv2.minAreaRect(np.array(contours_points))
        box = np.array(cv2.cv.BoxPoints(rect))
        boxes[id] = box
    return boxes
