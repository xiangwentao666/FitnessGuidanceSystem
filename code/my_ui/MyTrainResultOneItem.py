# -*- coding: utf-8 -*-
# @File : MyTrainResultOneItem.py 
# @Description : 
# @Author : Xiang Wentao, Software College of NEU
# @Contact : neu_xiangwentao@163.com
# @Date : 2023/3/13 11:49
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QListWidgetItem, QListWidget, QWidget

from ui.TrainResultOneItem import Ui_Form as uif

class MyTrainResultListItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, label, content):
        '''
        抓拍的列表中，每一个item的类，封装了每个item的ui、显示的数据等等
        :param list_widget:
        :param img:
        :param action_type:
        '''
        super(MyTrainResultListItem, self).__init__()
        self.list_widget = list_widget
        self.widget = MyTrainResultListItem.Widget(list_widget)
        # self.setSizeHint(QSize(150, 150))
        self.widget.lineEdit.setText(content)
        self.widget.lineEdit.setEnabled(False)
        self.widget.label.setText(label)
        self.widget.setFixedHeight(50)

    def add_item(self):
        # size = self.sizeHint()
        self.list_widget.insertItem(0, self)  # 添加
        # self.widget.setSizeIncrement(size.width(), 50)
        self.list_widget.setItemWidget(self, self.widget)

    class Widget(QWidget, uif):
        def __init__(self, parent=None):
            super(MyTrainResultListItem.Widget, self).__init__(parent)
            self.setupUi(self)


# class MyTrainResultOneItem(QListWidgetItem):
#     def __init__(self, listWidget, label_value: str, input_value: str, current_item_count: int):
#         super(MyTrainResultOneItem, self).__init__()
#         self.setupUi(self)
#         self.label.setText(label_value)
#         # print(label_value)
#         self.lineEdit.setText(input_value)
#         # print(input_value)
#         # print(current_item_count)
#
#     def addWidget(self):
#         self.listWidget.add
#
#     class Widget(QWidget, Ui_RealTimeCatch):
#         def __init__(self, parent=None):
#             super(CriticalFrameListItem.Widget, self).__init__(parent)
#             self.setupUi(self)