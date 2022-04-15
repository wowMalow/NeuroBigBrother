import itertools
import cv2
import mediapipe as mp
import math
import numpy as np
import utils as ut
import random
import os


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


def save_img(image: np.array, count: int, position: tuple, folder_name: str) -> None:
    x, y = position

    is_written = cv2.imwrite(img=image, filename=f"./data/{folder_name}/{ut.SESSION_ID}/{count:06d}_{x}_{y}.jpg")
    if not is_written:
        print('Error while saving image')


def update_train_screen(rect_size: int) -> tuple:
    width, height = ut.WIDTH, ut.HEIGHT
    size = rect_size

    black_screen = np.zeros((height, width))

    x = random.randint(size, width-size)
    y = random.randint(size, height-size)

    cv2.rectangle(black_screen, (x-size, y-size), (x+size, y+size), (255, 255, 255), -1)
    return x, y, black_screen


def main():
    screen_width, screen_height = 1920, 1080

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    caption = cv2.VideoCapture(0)
    caption.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    caption.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    window_name = 'Eyez'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen_width - 1, screen_height - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    mode = 'selfie'

    x_target, y_target, train_screen = update_train_screen(rect_size=5)

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

                try:
                    # Image centered by iris
                    left_eye_iris = crop_frame(mesh_coords, ut.LEFT_IRIS, frame, 3, 100)
                    right_eye_iris = crop_frame(mesh_coords, ut.RIGHT_IRIS, frame, 3, 100)

                    # Centered by eyelid
                    left_eye = crop_frame(mesh_coords, ut.LEFT_EYE, frame, 1.2, 100)
                    right_eye = crop_frame(mesh_coords, ut.RIGHT_EYE, frame, 1.2, 100)

                    both_eyes_iris = np.hstack((left_eye_iris, right_eye_iris))
                    both_eyes = np.hstack((left_eye, right_eye))

                    success_tracking = True
                except BaseException:
                    print('Eyes are uot of screen')
                    success_tracking = False

            match mode:
                case 'selfie':
                    img_show = frame
                case 'eyez':
                    img_show = both_eyes
                case 'train':
                    img_show = train_screen
                case _:
                    img_show = frame

            cv2.imshow(window_name, img_show)

            pressed_key = cv2.waitKey(1)
            if pressed_key == ord(' '):
                if (mode == 'train') & success_tracking:
                    count = next(ut.image_counter)
                    save_img(both_eyes_iris, count, (x_target, y_target), 'iris')
                    save_img(both_eyes, count, (x_target, y_target), 'eyelid')
                    x_target, y_target, train_screen = update_train_screen(rect_size=5)
            elif pressed_key == ord('s'):
                mode = 'selfie'
            elif pressed_key == ord('e'):
                mode = 'eyez'
            elif pressed_key == ord('t'):
                mode = 'train'
            elif pressed_key == ord('q'):
                break
    caption.release()


if __name__ == '__main__':
    main()