from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np
import imutils
import cv2
import math

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def extract_bboxes(fused):
    mask = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask < 40, 0, 1).astype(np.uint8)
    boxes = []

    contours_, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([y, x, y + h, x + w])
    
    if not boxes:
        boxes.append([0, 0, fused.shape[0], fused.shape[1]])

    return np.array(boxes[:1], dtype=np.int32)  # Only first box for now

def getContours(npImage, overlay_img, realHeight, realWidth, unit, confidence, angle_th=30):
    image = npImage.copy()
    imgHeight, imgWidth = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 80)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts, _ = contours.sort_contours(cnts)

    pixelsPerMetricHeight = realHeight / imgHeight
    pixelsPerMetricWidth = realWidth / imgWidth

    y1, x1, y2, x2 = extract_bboxes(npImage)[0]
    cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.line(overlay_img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 255, 0), 1)

        top_p = min([(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY))], key=lambda x: x[1])
        bot_p = max([(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY))], key=lambda x: x[1])
        D_ad = dist.euclidean(top_p, bot_p) + 1e-7

        P1 = min(top_p, bot_p, key=lambda x: x[0])
        P2 = max(top_p, bot_p, key=lambda x: x[0])
        slope = (P1[1] - P2[1]) / (P2[0] - P1[0]) if (P2[0] - P1[0]) != 0 else 0

        angle = np.arccos(abs(top_p[0] - bot_p[0]) / D_ad) * 180 / math.pi
        angle_text_pos = (top_p[0], top_p[1] + 15)
        cv2.putText(overlay_img, f"angle={angle:.1f}", angle_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        length = cv2.arcLength(c, True) / 2. * pixelsPerMetricWidth

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        mask = gray.copy()
        mask[mask < 40] = 0
        row = mask[cY]
        if not np.any(row):
            continue
        width = cv2.countNonZero(row)
        right_x = np.max(np.nonzero(row))
        left_x = np.min(np.nonzero(row))

        cv2.line(overlay_img, (left_x, cY), (right_x, cY), (255, 0, 255), 1)
        width *= pixelsPerMetricWidth

        if angle < angle_th:
            category = 'H'
            text_origin = (int(tltrX), int(tltrY) + 40)
        else:
            category = 'V'
            text_origin = (int(tltrX), int(tltrY))

        direction = 'L' if slope > 0 else 'R'
        category += direction

        cv2.putText(overlay_img, f"L={length:.1f}{unit}", text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(overlay_img, f"W={width:.1f}{unit}", (text_origin[0], text_origin[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(overlay_img, f"Crack {confidence.item() * 100:.2f}% cat={category}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (36, 255, 12), 1)

    return overlay_img
