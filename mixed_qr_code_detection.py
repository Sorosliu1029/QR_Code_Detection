# coding: utf-8
import cv2
import numpy as np
import qr_utils as qr


def main(image_path):
    qr_origin, qr_gray = qr.read_image(image_path)
    qr.show(qr_origin)
    qr.show(qr_gray)
    edges = qr.get_edges(qr_origin)
    qr.show(edges)
    contours, hierarchy = qr.get_contours(edges)
    found = qr.get_position_patterns(contours, hierarchy)
    qr.draw_separate_position_patterns(qr_origin, found, contours)
    contours_points = qr.get_contours_points(found, contours)
    box = qr.get_area_box(contours_points)
    qr.draw_box(qr_origin, box)
    qr.draw_all_position_patterns(qr_origin, found, contours)
    boxes = qr.get_boxes(found, contours)
    qr.draw_lines(qr_origin, boxes)
    th, qr_bi = qr.get_binary_image(qr_gray)
    boxes_dict, timing_patterns = qr.get_valid_boxes_index(boxes, qr_bi)
    print boxes_dict
    qr.draw_timing_pattern(qr_origin, timing_patterns)
    qr_code_boxes = qr.get_qr_code_boxes(boxes_dict, contours, found)
    draw_img = qr_origin.copy()
    for qr_code_box in qr_code_boxes.values():
        cv2.polylines(draw_img, np.int32([qr_code_box]), True, (255, 0, 0), 4)
    qr.show(draw_img)

if __name__ == '__main__':
    main('two_qrs_cut2.png')
    main('two_qrs_cut1.png')
    main('qr_code_mixed.png')
    main('two_qrs.png')
