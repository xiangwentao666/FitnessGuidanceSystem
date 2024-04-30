# -*- coding: utf-8 -*-
import datetime

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QDesktopWidget

from ui.video_play_window import Ui_MainWindow


class My_Ui_MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self, p=None):
        super().__init__()
        self.setupUi(self)
        self.video_path = p
        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture(0)

    def set_window_title(self, p):
        self.setWindowTitle(p)

    def set_video_path(self, p):
        self.video_path = p
        self.cap = cv2.VideoCapture(p)

    def start_to_play(self):
        self.show()
        self.timer_camera.start(1000/self.cap.get(cv2.CAP_PROP_FPS)*5)
        # self.timer_camera.start(1000/cv2.GetCaptureProperty(self.cap, cv2.CV_CAP_PROP_FPS))
        self.timer_camera.timeout.connect(self.open_frame)

    def open_frame(self):
        ret, image = self.cap.read()
        if ret:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            elif len(image.shape) == 1:
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
            else:
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.video_play_area_label.setPixmap(QPixmap(vedio_img))
            self.video_play_area_label.setScaledContents(True)  # 自适应窗口
        else:
            self.hide()
            self.cap.release()
            self.timer_camera.stop()

    def stop_play(self):
        self.hide()
        self.cap.release()
        self.timer_camera.stop()

    def closeEvent(self, event):
        self.stop_play()
        self.releaseAll()
        self.close()
        event.accept()

    def releaseAll(self):
        self.cap.release()
