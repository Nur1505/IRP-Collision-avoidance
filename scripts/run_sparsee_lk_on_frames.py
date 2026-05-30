#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np


INPUT_DIR = "../extracted_frames"
OUTPUT_DIR = "../lk_sparse_results"

FRAME_SKIP = 2
RESIZE_TO = (640, 480)

FEATURE_PARAMS = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


def draw_sparse_flow(frame, old_pts, new_pts, status):
    output = frame.copy()

    good_old = old_pts[status.flatten() == 1]
    good_new = new_pts[status.flatten() == 1]

    for old, new in zip(good_old, good_new):
        x_old, y_old = old.ravel()
        x_new, y_new = new.ravel()

        cv2.arrowedLine(
            output,
            (int(x_old), int(y_old)),
            (int(x_new), int(y_new)),
            (0, 255, 0),
            2,
            tipLength=0.3
        )

        cv2.circle(output, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

    return output, len(good_new)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

    if len(frame_paths) < 2:
        raise RuntimeError("Not enough frames found in extracted_frames.")

    for idx in range(0, len(frame_paths) - FRAME_SKIP, FRAME_SKIP):
        path1 = frame_paths[idx]
        path2 = frame_paths[idx + FRAME_SKIP]

        frame1 = cv2.imread(path1)
        frame2 = cv2.imread(path2)

        if frame1 is None or frame2 is None:
            continue

        frame1 = cv2.resize(frame1, RESIZE_TO)
        frame2 = cv2.resize(frame2, RESIZE_TO)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        old_pts = cv2.goodFeaturesToTrack(gray1, mask=None, **FEATURE_PARAMS)

        if old_pts is None:
            print("No features found in", path1)
            continue

        new_pts, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            old_pts,
            None,
            **LK_PARAMS
        )

        if new_pts is None or status is None:
            print("Tracking failed for", path1)
            continue

        flow_img, tracked_count = draw_sparse_flow(frame2, old_pts, new_pts, status)

        cv2.putText(
            flow_img,
            "Tracked features: {}".format(tracked_count),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        out_path = os.path.join(OUTPUT_DIR, "sparse_flow_{:06d}.png".format(idx))
        cv2.imwrite(out_path, flow_img)

        print("Saved:", out_path, "| tracked features:", tracked_count)

    print("Done. Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()