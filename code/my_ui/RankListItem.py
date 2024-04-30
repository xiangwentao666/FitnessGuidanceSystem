# -*- coding: utf-8 -*-
from PyQt5.QtCore import QSize, QModelIndex, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QAbstractItemView, QListWidget, QWidget, QHBoxLayout, QVBoxLayout, QLabel, \
    QListWidgetItem

class RankListItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, data:dict):
        '''
        :param list_widget:
        :param data: 单条视频记录的dict格式的数据
        '''
        super(RankListItem, self).__init__()
        self.list_widget = list_widget
        self.nickname = data['nickname']
        self.finish_seconds = data['finish_seconds']

    def add_item(self):
        def get_item_wight(nickname, finish_seconds):
            # 总Widget
            wight = QWidget()
            # 总体横向布局
            layout_main = QHBoxLayout()
            # 纵向布局
            # 右下的的横向布局
            f = QFont('微软雅黑', 10, QFont.Bold)
            # f = QFont('times', 7, QFont.Black, QFont.Bold)
            layout_right_down = QHBoxLayout()  # 右下的横向布局
            pth_qlabel = QLabel("{}".format(nickname))
            pth_qlabel.setFont(f)
            layout_right_down.addWidget(pth_qlabel)
            pth_qlabel1 = QLabel("{}秒".format(finish_seconds))
            f = QFont('微软雅黑', 12, QFont.Bold)
            pth_qlabel1.setFont(f)
            layout_right_down.addWidget(pth_qlabel1)
            # 按照从左到右, 从上到下布局添加
            layout_main.addLayout(layout_right_down)  # 右边的布局
            wight.setLayout(layout_main)  # 布局给wight
            return wight  # 返回wight
        self.setSizeHint(QSize(200, 80))  # 设置QListWidgetItem大小
        widget = get_item_wight(self.nickname, self.finish_seconds)  # 调用上面的函数获取对应
        self.list_widget.insertItem(0, self)  # 添加item
        self.list_widget.setItemWidget(self, widget)  # 为item设置widget
