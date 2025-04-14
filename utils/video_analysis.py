import cv2
import numpy as np
import os

def analyze_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_diffs = []
        last_frame = None
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if last_frame is not None:
                diff = cv2.absdiff(last_frame, gray)
                mean_diff = np.mean(diff)
                frame_diffs.append(mean_diff)

            last_frame = gray
            count += 1
            if count > 50:
                break

        cap.release()

        avg_motion = np.mean(frame_diffs)
        if avg_motion < 1:
            return "⚠️ Possible static tampering detected (avg motion diff: {:.2f})".format(avg_motion)
        elif np.std(frame_diffs) > 20:
            return "⚠️ Possible inconsistent motion — suspect manipulation (std dev: {:.2f})".format(np.std(frame_diffs))
        else:
            return "✅ Video appears to be real (avg motion diff: {:.2f})".format(avg_motion)

    except Exception as e:
        return f"Error analyzing video: {str(e)}"
