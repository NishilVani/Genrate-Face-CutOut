import cv2
import mediapipe as mp
import numpy as np
from math import degrees, atan, hypot
from PIL import Image


mp_selfie = mp.solutions.selfie_segmentation
mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.75)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
cap = cv2.VideoCapture(0)

# Create with statement for model
with mp_selfie.SelfieSegmentation(model_selection=0) as model:
    while True:
        success, frame = cap.read()
        cv2.imshow('Real', frame)
        bbox = False
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:

                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height, width, alpha = frame.shape

                # Toppest Point On Face
                pt = faceLms.landmark[10]
                x1 = int(pt.x * width)
                y1 = int(pt.y * height)

                # Lowest Point On Face
                pt = faceLms.landmark[152]
                x2 = int(pt.x * width)
                y2 = int(pt.y * height)

                # Right Point
                pt = faceLms.landmark[234]
                x3 = int(pt.x * width)
                y3 = int(pt.y * height)

                # left Point
                pt = faceLms.landmark[454]
                x4 = int(pt.x * width)
                y4 = int(pt.y * height)

                # Head Center
                pt = faceLms.landmark[8]
                x5 = int(pt.x * width)
                y5 = int(pt.y * height)
                
                # Bottom Ear
                pt = faceLms.landmark[361]
                x7 = int(pt.x * width)
                y7 = int(pt.y * height)

                pt = faceLms.landmark[58]
                x6 = int(pt.x * width)
                y6 = int(pt.y * height)

                landmarks_points = []
                for n in range(0, 468):
                    pt = faceLms.landmark[n]
                    x = int(pt.x * width)
                    y = int(pt.y * height)
                    landmarks_points.append((x, y))


                points = ((x6, y6), (x7, y7), (0, y6), (width, y7), (width, height), (0, height))
                points1 = np.array(points, np.int32)
                points1 = points1.astype(np.int32)
                convexhull = cv2.convexHull(points1)

                # Apply segmentation
                
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                res = model.process(frame)
                frame.flags.writeable = True

                ih, iw, ic = frame.shape

                background = np.zeros(frame.shape, dtype=np.uint8)
                mask = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.5

                mask = mask.astype(np.int32)
                cv2.fillConvexPoly(mask, convexhull, (0, 0, 0))

                points = np.array(landmarks_points, np.int32)
                convexhull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, convexhull, (255, 255, 255))

                frame = np.where(mask, frame, background)
                # Angle
                if x4 != x3:
                    angle = degrees(atan((y4 - y3) / (x4 - x3)))
                    if y1 > y2:
                        angle = 180 + angle
                    im1 = Image.fromarray(frame)
                    im1 = im1.rotate(angle, expand=True)
                    frame = np.array(im1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow('Cut-Out', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
