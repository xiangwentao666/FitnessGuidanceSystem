# -*- coding: utf-8 -*-
import datetime
import json
import os.path
import threading

import requests
from PyQt5.QtCore import QUrl, pyqtSignal, QSize
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QListWidget, QWidget, QHBoxLayout, QVBoxLayout, QLabel, \
    QAbstractItemView
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QDesktopWidget
import pickle

from ui.history_grade_item import Ui_Form

class HistoryGradeListItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, data:dict):
        '''
        :param list_widget:
        :param data: 单条成绩记录的dict格式的数据
        '''
        super(HistoryGradeListItem, self).__init__()
        self.list_widget = list_widget
        self.data = data
        # self.widget = HistoryGradeListItem.Widget(data, parent=list_widget)

    def add_item(self):
        def get_item_wight(data):
            # 读取属性
            # {'train_plan_name': '肱二头肌训练',
            # 'finish_count': '4/4',
            # 'seconds_last': '52',
            #  'finish_datetime': '2023-05-08 14:17:47'}
            train_plan_name = data['train_plan_name']
            datetimeee = data['finish_datetime']
            finish_count = data['finish_count']
            seconds_last = data['seconds_last']
            # 总Widget
            wight = QWidget()

            # 总体横向布局
            layout_main = QHBoxLayout()
            # 纵向布局
            layout_right = QVBoxLayout()
            # 右下的的横向布局
            layout_right_down = QHBoxLayout()  # 右下的横向布局
            layout_right_bottom = QHBoxLayout()  # 右下的横向布局
            layout_right_down.addWidget(QLabel("·完成个数：{}".format(finish_count if type(finish_count)!=tuple else finish_count[0])))
            layout_right_down.addWidget(QLabel("·剩余时间：{}".format(seconds_last if type(seconds_last)!=tuple else seconds_last[0])))

            layout_right_bottom.addWidget(QLabel("".join(["-"*200])))

            # 按照从左到右, 从上到下布局添加
            layout_right.addWidget(QLabel("【名称】：{}".format(train_plan_name if type(train_plan_name)!=tuple else train_plan_name[0]))) # 右边的纵向布局
            layout_right.addWidget(QLabel("【日期】：{}".format(datetimeee if type(datetimeee)!=tuple else datetimeee[0]))) # 右边的纵向布局
            layout_right.addLayout(layout_right_down)  # 右下角横向布局
            layout_right.addLayout(layout_right_bottom)  # 右下角横向布局
            layout_main.addLayout(layout_right)  # 右边的布局

            wight.setLayout(layout_main)  # 布局给wight
            return wight  # 返回wight

        item = QListWidgetItem()  # 创建QListWidgetItem对象
        item.setSizeHint(QSize(200, 100))  # 设置QListWidgetItem大小
        widget = get_item_wight(self.data)  # 调用上面的函数获取对应
        # self.list_widget.insertItem(0, item)  # 添加item
        self.list_widget.addItem(item)  # 添加item
        self.list_widget.setItemWidget(item, widget)  # 为item设置widget

    class Widget(QWidget, Ui_Form):
        def __init__(self, data, parent=None):
            super(HistoryGradeListItem.Widget, self).__init__(parent)
            self.setupUi(self)
            self.username_label.setText("")
