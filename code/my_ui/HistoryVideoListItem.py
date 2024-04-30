# -*- coding: utf-8 -*-
import json
import threading
from os import listdir as os_listdir, system as os_system
from os.path import exists as os_path_exists, basename as os_path_basename, abspath as os_path_abspath

import cv2
import requests
from PyQt5.QtCore import QSize, QModelIndex, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QAbstractItemView, QListWidget, QWidget, QHBoxLayout, QVBoxLayout, QLabel, \
    QListWidgetItem

class HistoryVideoListItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, data:dict):
        '''
        :param list_widget:
        :param data: 单条视频记录的dict格式的数据
        '''
        super(HistoryVideoListItem, self).__init__()
        self.list_widget = list_widget
        self.video_file_path = data['file_path']
        self.uploader_username = data['username']
        self.upload_datetime = data['datetime']
        self.video_file_path_to_display = data['file_path']

    def get_video_file_path(self):
        return self.video_file_path

    def add_item(self):
        def get_item_wight(pth_toshow, un, dt):
            # 总Widget
            wight = QWidget()

            # 总体横向布局
            layout_main = QHBoxLayout()
            # 纵向布局
            layout_right = QVBoxLayout()
            # 右下的的横向布局
            f = QFont('微软雅黑', 7, QFont.Bold)
            # f = QFont('times', 7, QFont.Black, QFont.Bold)
            layout_right_down = QHBoxLayout()  # 右下的横向布局
            pth_qlabel = QLabel("路径：{}".format(pth_toshow if type(pth_toshow) != tuple else pth_toshow[0]))
            pth_qlabel.setFont(f)
            layout_right_down.addWidget(pth_qlabel)

            # 按照从左到右, 从上到下布局添加
            username_qlabel = QLabel("名称：{}".format(un if type(un)!=tuple else un[0]))
            username_qlabel.setFont(f)
            layout_right.addWidget(username_qlabel) # 右边的纵向布局
            datetime_qlabel = QLabel("日期：{}".format(dt if type(dt) != tuple else dt[0]))
            datetime_qlabel.setFont(f)
            layout_right.addWidget(datetime_qlabel) # 右边的纵向布局
            layout_right.addLayout(layout_right_down)  # 右下角横向布局
            layout_main.addLayout(layout_right)  # 右边的布局

            wight.setLayout(layout_main)  # 布局给wight
            return wight  # 返回wight
        self.setSizeHint(QSize(200, 80))  # 设置QListWidgetItem大小
        widget = get_item_wight(self.video_file_path_to_display, self.uploader_username, self.upload_datetime)  # 调用上面的函数获取对应
        self.list_widget.insertItem(1, self)  # 添加item
        self.list_widget.setItemWidget(self, widget)  # 为item设置widget
