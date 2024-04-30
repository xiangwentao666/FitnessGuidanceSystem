# -*- coding: utf-8 -*-
# @File : MyTrainResult.py 
# @Description : 
# @Author : Xiang Wentao, Software College of NEU
# @Contact : neu_xiangwentao@163.com
# @Date : 2023/3/13 11:49
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog

from ui.TrainResult import Ui_Form as uif
from my_ui.MyTrainResultOneItem import MyTrainResultListItem


class MyTrainResult_Ui_Form(uif, QtWidgets.QWidget):
    def __init__(self, train_result_signal: pyqtSignal, clear_camera_label_signal: pyqtSignal):
        super(MyTrainResult_Ui_Form, self).__init__()
        self.setupUi(self)
        self.clear_camera_label_signal = clear_camera_label_signal
        train_result_signal.connect(self.train_result_signal_emit_handler)
        self.setFixedSize(410, 330)

    def train_result_signal_emit_handler(self, train_result: dict):
        print("receive: {}".format(train_result))
        MyTrainResultListItem(
                self.listWidget,
                '训练项目：',
                train_result['name']
        ).add_item()
        MyTrainResultListItem(
            self.listWidget,
            '完成个数：',
                train_result['finish_count']
        ).add_item()
        MyTrainResultListItem(
            self.listWidget,
            '剩余时间：',
            "{}s".format(train_result['seconds_last'])
        ).add_item()
        MyTrainResultListItem(
            self.listWidget,
            '结束时间：',
            train_result['finish_datetime']
        ).add_item()

    def clear_items(self):
        self.listWidget.clear()

    def closeEvent(self, event):
        self.clear_items()
        self.clear_camera_label_signal.emit()
        event.accept()