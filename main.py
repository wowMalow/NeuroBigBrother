import itertools

import cv2
import mediapipe as mp
import math
import numpy as np
import utils as ut




def landmarks_detection(img, results):
    height, width = img.shape[:2]
    mesh_coord = [(int(point.x * width), int(point.y * height))
                  for point in results.multi_face_landmarks[0].landmark]
    return mesh_coord


def crop_frame(mesh: list, edge: tuple, image, scale: float, size: int):
    x_start, y_start = mesh[edge[0]]
    x_end, y_end = mesh[edge[1]]

    dx = x_end - x_start
    dy = y_end - y_start
    x_center = x_start + dx // 2
    y_center = y_start + dy // 2

    margin = int(scale * math.sqrt(dx**2 + dy**2) // 2)

    cropped_img = image[y_center-margin: y_center+margin, x_center-margin: x_center+margin]
    resized_img = cv2.resize(cropped_img, (size, size))
    return resized_img


def save_img(image: np.array, counter: itertools.count) -> None:
    count = next(counter)
    is_written = cv2.imwrite(img=image, filename=f"./data/{ut.SESSION_ID}/{count:06d}.jpg")
    if not is_written:
        print('Error while saving image')


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

caption = cv2.VideoCapture(0)
caption.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
caption.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
    while caption.isOpened():
        success, frame = caption.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Do if you need selfie-like view
        # To improve performance, optionally mark the image as not writeable
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            mesh_coords = landmarks_detection(frame, results)

            # left_eye = crop_frame(mesh_coords, ut.LEFT_IRIS, frame, 3, 100)
            # right_eye = crop_frame(mesh_coords, ut.RIGHT_IRIS, frame, 3, 100)
            left_eye = crop_frame(mesh_coords, ut.LEFT_EYE, frame, 1.2, 100)
            right_eye = crop_frame(mesh_coords, ut.RIGHT_EYE, frame, 1.2, 100)

            both_eyes = np.hstack((left_eye, right_eye))

        cv2.imshow('Eyez', both_eyes)

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord(' '):
            save_img(both_eyes, ut.image_counter)
        elif pressed_key == ord('q'):
            break
caption.release()