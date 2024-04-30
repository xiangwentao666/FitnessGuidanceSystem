# -*- coding: utf-8 -*-
# @File : main.py
# @Description :
# @Author : Xiang Wentao, Software College of NEU
# @Contact : neu_xiangwentao@163.com
# @Date : 2023/2/24 16:10
import sys

from PyQt5.QtWidgets import QApplication

from my_ui.MyMainWindow import My_Ui_MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # from QcureUi import cure
    window = My_Ui_MainWindow()
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
