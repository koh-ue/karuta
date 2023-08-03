#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: FIXME: XXX: HACK: NOTE: INTENT: USAGE:

import os
import re

import sys
import json
#import joblib
import argparse
import functools
import itertools
import cv2 as cv
import numpy as np
#import pandas as pd
#import seaborn as sns
#from collections import defaultdict

sys.path.append(".")

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("-d", "--device_id", type=int, default="1", help="iMac Camera:0, iPhone Camera:1")
#parser.add_argument("--seed", type=int, required=True)
#parser.add_argument("--distance", type=float, default=42.195)
#parser.add_argument("--condition", type=str, choices=["sunny", "rainy", "cloudy"], default="sunny")
#parser.add_argument("--is_competition", action="store_true")

args = parser.parse_args()

DEVICE_ID = args.device_id

# NOTE: AREA for functions.

def view_camera(device_id):
    cap = cv.VideoCapture(device_id)

    templates = []
    raw_patterns = os.listdir('patterns')
    for pattern in raw_patterns:
        if pattern.endswith(".jpg"):
            tmp = cv.imread(f"patterns/{pattern}", cv.IMREAD_GRAYSCALE)
            tmp = cv.resize(tmp, (85, 85), cv.INTER_NEAREST)
            flipped_tmp = cv.flip(tmp, -1)
            song_number = int(os.path.splitext(pattern)[0])

            templates.append((tmp, flipped_tmp, song_number))

    method = cv.TM_CCOEFF_NORMED
    threshold = 0.35
    fontFamily = cv.FONT_HERSHEY_PLAIN
    fontSize = 7
    textColor = (252, 71, 37)
    thick = 5
    linetype = cv.LINE_AA

    with open('kimariji.txt', 'r') as f:
        kimariji = f.readlines()
    for i in range(len(kimariji)):
        kimariji[i] = kimariji[i].replace("\n", "")
            
    print(kimariji)
    while(cap.isOpened()):

        _, frame = cap.read()
        

        lower = np.array([30, 64, 0])
        upper = np.array([90, 255, 255])
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_mask = cv.inRange(hsv, lower, upper)

        green_area = cv.bitwise_and(frame, frame, mask=hsv_mask)
        green_area_gray_scale = cv.cvtColor(green_area, cv.COLOR_BGR2GRAY)
        _, threshed_img = cv.threshold(green_area_gray_scale, 70, 255, cv.THRESH_BINARY)
        cv.imshow("Destination", threshed_img)

        contours, hierarchy = cv.findContours(
            threshed_img, 
            cv.RETR_EXTERNAL,      # 一番外側の輪郭のみを取得する 
            cv.CHAIN_APPROX_NONE   # 輪郭座標の省略なし
        )

        for i, contour in enumerate(contours):
            # 輪郭を描画
            #cv.drawContours(frame, contours, i, (255, 0, 0), 2)
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.intp(box)

            center, size, angle = rect
            center = tuple(map(int, center))  # float -> int
            size = tuple(map(int, size))  # float -> int
            frame_h, frame_w = frame.shape[:2]  # 画像の高さ、幅
            area = size[0] * size[1]

            if area > 9000:
                cv.drawContours(frame,[box],0,(0,0,255), 2)

                M = cv.getRotationMatrix2D(center, angle, 1)
                rotated = cv.warpAffine(frame, M, (frame_w, frame_h))
                cropped = cv.getRectSubPix(rotated, size, center)
                if cropped.shape[0] < cropped.shape[1]:
                    cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)

                target = cv.cvtColor(cropped, cv.COLOR_RGB2GRAY)
                target = cv.resize(target, (100, 100), cv.INTER_NEAREST)

                for template_pack in templates:
                    two_templates = [template_pack[0], template_pack[1]]
                    template_number = template_pack[2]

                    results = []
                    for template in two_templates:
                        res = cv.matchTemplate(target, template, method)
                        _, max_val, _, max_loc = cv.minMaxLoc(res)
                        results.append([max_val, max_loc])

                    true_max_val = max([results[0][0], results[1][0]])
                    true_max_index = [results[0][0], results[1][0]].index(true_max_val)
                    print(true_max_val)

                    w, h = two_templates[true_max_index].shape[::-1]
                    top_left = results[true_max_index][1]
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv.rectangle(target, top_left, bottom_right, 255, 2)
                    if true_max_val > threshold:
                        print(f"No.{template_number} Detected!")

                        text = kimariji[int(template_number)-1]
                        size, _ = cv.getTextSize(text, fontFace=fontFamily, fontScale=fontSize, thickness=thick)
                        text_width, text_height = size
                        cv.putText(frame, text, (center[0] - int(text_width/2), center[1] + int(text_height/2)), fontFace=fontFamily, fontScale=fontSize, color=textColor, thickness=thick, lineType=linetype)

                        cv.imshow(f"No.{template_number}", target)

                
            
        cv.imshow("Raw", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return 0

if __name__ == '__main__':
    print("hellow world!")
    print(cv.getBuildInformation())
    view_camera(DEVICE_ID)