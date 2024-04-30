# -*- coding: utf-8 -*-
# @File : MyMediaPipe.py
# @Description : 
# @Author : Xiang Wentao, Software College of NEU
# @Contact : neu_xiangwentao@163.com
# @Date : 2023/3/16 20:04
import json

import cv2
import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
#人体姿态编码
class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

class MyMediaPipe():
    _landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    KP_NOSE = "nose"
    KP_RIGHT_EYE_INNER = "right_eye_inner"
    KP_RIGHT_EYE = "right_eye"
    KP_RIGHT_EYE_OUTER = "right_eye_outer"
    KP_LEFT_EYE_INNER = "left_eye_inner"
    KP_LEFT_EYE = "left_eye"
    KP_LEFT_EYE_OUTER = "left_eye_outer"
    KP_RIGHT_EAR = "right_ear"
    KP_LEFT_EAR = "left_ear"
    KP_MOUTH_RIGHT = "mouth_right"
    KP_MOUTH_LEFT = "mouth_left"
    KP_RIGHT_SHOULDER = "right_shoulder"
    KP_LEFT_SHOULDER = "right_shoulder"
    KP_RIGHT_ELBOW = "right_shoulder"
    KP_LEFT_ELBOW = "right_shoulder"
    KP_RIGHT_WRIST = "right_shoulder"
    KP_LEFT_WRIST = "right_shoulder"
    KP_RIGHT_PINKY = "right_shoulder"
    KP_LEFT_PINKY = "right_shoulder"
    KP_LEFT_INDEX = "right_shoulder"
    KP_RIGHT_INDEX = "right_shoulder"
    KP_LEFT_THUMB = "right_shoulder"
    KP_RIGHT_THUMB = "right_shoulder"
    KP_LEFT_HIP = "right_shoulder"
    KP_RIGHT_HIP = "right_shoulder"
    KP_LEFT_KNEE = "right_shoulder"
    KP_RIGHT_KNEE = "right_shoulder"
    KP_RIGHT_HIP = "right_shoulder"
    KP_RIGHT_HIP = "right_shoulder"
    KP_RIGHT_HIP = "right_shoulder"
    KP_RIGHT_HIP = "right_shoulder"

    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.width = 500
        self.height = 300
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=True,
            smooth_landmarks=True,
            # model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.flag = False
        # self.cp = cv2.VideoCapture(0)

    def clear_drawed_image(self):
        self.drawed_image = None

    def get_drawed_image(self):
        try:
            self.mp_drawing.draw_landmarks(
                self.drawed_image, self.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            return self.drawed_image
        except:
            return None

    def is_right_elbow_lower_than_right_wrist(self):
        re_pos = self.get_pos("right_elbow")
        rw_pos = self.get_pos("right_wrist")
        if re_pos is None or rw_pos is None:
            return None
        if re_pos[1] < rw_pos[1]:
            return True
        else:
            return False

    def is_right_elbow_higher_than_right_wrist(self):
        re_pos = self.get_pos("right_elbow")
        rw_pos = self.get_pos("right_wrist")
        if re_pos is None or rw_pos is None:
            return None
        if re_pos[1] > rw_pos[1]:
            return True
        else:
            return False

    def get_pos(self, joint):
        return self.landmarks[self._landmark_names.index(joint)] if self.landmarks is not None else None

    def get_landmarks(self):
        return self.landmarks

    def process_one_frame_and_parse_landmarks(self, raw_frame1):
        '''
        raw_frame1
        :param raw_frame1:
        :return:
        '''
        raw_frame1 = raw_frame1.astype(np.uint8)
        raw_frame = cv2.cvtColor(raw_frame1, cv2.COLOR_BGR2RGB)
        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            # 对获取的图片转换成二维矩阵形式，后再将RGB转成BGR
            # 因为imshow,默认通道顺序是BGR，而pyautogui默认是RGB所以要转换一下，不然会有点问题
            image = cv2.cvtColor(np.asarray(raw_frame), cv2.COLOR_RGB2BGR)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # results = holistic.process(image)
            results = self.pose.process(image)
            pose_landmarks = results.pose_landmarks
            self.drawed_image = image
            self.pose_landmarks = pose_landmarks
            if pose_landmarks is not None:
                land_mark = pose_landmarks.landmark
                # 横着向右为x轴正向
                # 竖着向下为y轴正向
                # 获取32个人体关键点坐标, index记录是第几个关键点
                self.landmarks = [[index, lm.x, lm.y, lm.visibility] for index, lm in enumerate(results.pose_landmarks.landmark)]
            else:
                self.landmarks = None

    def monitor(self):
        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

            has_frame, image = self.cp.read()
            while True:
                # 对获取的图片转换成二维矩阵形式，后再将RGB转成BGR
                # 因为imshow,默认通道顺序是BGR，而pyautogui默认是RGB所以要转换一下，不然会有点问题
                image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                # cv2.imshow("截屏", image)
                # cv2.waitKey(5)

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # results = holistic.process(image)
                results = self.pose.process(image)
                pose_landmarks = results.pose_landmarks
                if pose_landmarks is not None:
                    land_mark = pose_landmarks.landmark
                    if not self.flag:
                        print(land_mark)
                        self.flag = not self.flag

                    # 横着向右为x轴正向
                    # 竖着向下为y轴正向

                    # 获取头部所有关键点的坐标
                    head_point_list = []
                    for i in range(4):
                        # 下标为0~10的点为头部的点
                        head_point_list.append(land_mark[i])

                    # 获取32个人体关键点坐标, index记录是第几个关键点
                    for index, lm in enumerate(results.pose_landmarks.landmark):
                        # 保存每帧图像的宽、高、通道数
                        h, w, c = image.shape

                        # 得到的关键点坐标x/y/z/visibility都是比例坐标，在[0,1]之间
                        # 转换为像素坐标(cx,cy)，图像的实际长宽乘以比例，像素坐标一定是整数
                        cx, cy = int(lm.x * w), int(lm.y * h)

                    # print(len(head_point_list))
                    visib_list = [_.visibility for _ in head_point_list]
                    max_index = visib_list.index(max(visib_list))

                    # print(max_index)
                    # print(head_point_list[max_index])

                    liable_point = head_point_list[max_index]  # 这里应换成置信度最高的一个点
                    nose_location_pix_x, nose_location_pix_y = int(liable_point.x * self.width), int(
                        liable_point.y * self.height)
                    # ↑ x 和 y 是在截图区域的图片中的像素点坐标，还需要转换为屏幕的坐标

                # 画图
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # mp_drawing.draw_landmarks(
                #     image,
                #     results.face_landmarks,
                #     mp_holistic.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    # self.mp_holistic.FACE_CONNECTIONS,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                        .get_default_pose_landmarks_style())

                # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Holistic', image)
                cv2.waitKey(5)
                has_frame, image = self.cp.read()

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

if __name__ == "__main__":
    my_mediapipe = MyMediaPipe()
    videoCapture = cv2.VideoCapture(0)
    hasFrame, frame = videoCapture.read()
    is_begin = False
    cnt = 0
    while hasFrame:
        my_mediapipe.process_one_frame_and_parse_landmarks(frame)
        print("invalid")
        hasFrame, frame = videoCapture.read()

