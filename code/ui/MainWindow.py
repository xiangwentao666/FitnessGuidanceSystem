# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1455, 665)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.analysing_block_tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.analysing_block_tabWidget.setGeometry(QtCore.QRect(10, 10, 1191, 601))
        self.analysing_block_tabWidget.setTabBarAutoHide(False)
        self.analysing_block_tabWidget.setObjectName("analysing_block_tabWidget")
        self.training_tab = QtWidgets.QWidget()
        self.training_tab.setObjectName("training_tab")
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.training_tab)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(10, 10, 1161, 551))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setSpacing(6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.train_plan_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.train_plan_label.setObjectName("train_plan_label")
        self.horizontalLayout_8.addWidget(self.train_plan_label)
        self.train_plan_comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.train_plan_comboBox.setObjectName("train_plan_comboBox")
        self.horizontalLayout_8.addWidget(self.train_plan_comboBox)
        self.horizontalLayout_8.setStretch(1, 5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.current_datetime_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.current_datetime_label.setObjectName("current_datetime_label")
        self.horizontalLayout_4.addWidget(self.current_datetime_label)
        self.current_datetime_value_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.current_datetime_value_label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.current_datetime_value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.current_datetime_value_label.setObjectName("current_datetime_value_label")
        self.horizontalLayout_4.addWidget(self.current_datetime_value_label)
        self.horizontalLayout_4.setStretch(1, 5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.count_down_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.count_down_label.setObjectName("count_down_label")
        self.horizontalLayout_5.addWidget(self.count_down_label)
        self.count_down_value_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.count_down_value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.count_down_value_label.setObjectName("count_down_value_label")
        self.horizontalLayout_5.addWidget(self.count_down_value_label)
        self.horizontalLayout_5.setStretch(1, 5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.finish_percentage_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.finish_percentage_label.setObjectName("finish_percentage_label")
        self.horizontalLayout_6.addWidget(self.finish_percentage_label)
        self.finish_percentage_value_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.finish_percentage_value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.finish_percentage_value_label.setObjectName("finish_percentage_value_label")
        self.horizontalLayout_6.addWidget(self.finish_percentage_value_label)
        self.horizontalLayout_6.setStretch(1, 5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.progressBar = QtWidgets.QProgressBar(self.horizontalLayoutWidget_7)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.verticalLayout_3.setStretch(2, 4)
        self.verticalLayout_3.setStretch(3, 4)
        self.horizontalLayout_7.addLayout(self.verticalLayout_3)
        self.train_display_area_verticalLayout = QtWidgets.QVBoxLayout()
        self.train_display_area_verticalLayout.setObjectName("train_display_area_verticalLayout")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.train_display_area_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.train_display_area_label.setObjectName("train_display_area_label")
        self.horizontalLayout_10.addWidget(self.train_display_area_label)
        self.monitor_hear_rate_comboBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_7)
        self.monitor_hear_rate_comboBox.setChecked(True)
        self.monitor_hear_rate_comboBox.setObjectName("monitor_hear_rate_comboBox")
        self.horizontalLayout_10.addWidget(self.monitor_hear_rate_comboBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem)
        self.train_display_area_verticalLayout.addLayout(self.horizontalLayout_10)
        self.train_display_area = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.train_display_area.setAutoFillBackground(False)
        self.train_display_area.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.train_display_area.setFrameShadow(QtWidgets.QFrame.Plain)
        self.train_display_area.setText("")
        self.train_display_area.setObjectName("train_display_area")
        self.train_display_area_verticalLayout.addWidget(self.train_display_area)
        self.train_display_area_button_horizontalLayout = QtWidgets.QHBoxLayout()
        self.train_display_area_button_horizontalLayout.setObjectName("train_display_area_button_horizontalLayout")
        self.train_display_area_realtime_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.train_display_area_realtime_button.setObjectName("train_display_area_realtime_button")
        self.train_display_area_button_horizontalLayout.addWidget(self.train_display_area_realtime_button)
        self.train_display_area_play_video_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.train_display_area_play_video_button.setObjectName("train_display_area_play_video_button")
        self.train_display_area_button_horizontalLayout.addWidget(self.train_display_area_play_video_button)
        self.train_display_area_stop_video_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.train_display_area_stop_video_button.setObjectName("train_display_area_stop_video_button")
        self.train_display_area_button_horizontalLayout.addWidget(self.train_display_area_stop_video_button)
        self.train_display_area_verticalLayout.addLayout(self.train_display_area_button_horizontalLayout)
        self.train_display_area_verticalLayout.setStretch(1, 10)
        self.train_display_area_verticalLayout.setStretch(2, 1)
        self.horizontalLayout_7.addLayout(self.train_display_area_verticalLayout)
        self.heart_rate_monitor_verticalLayout = QtWidgets.QVBoxLayout()
        self.heart_rate_monitor_verticalLayout.setObjectName("heart_rate_monitor_verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.heart_rate_monitor_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.heart_rate_monitor_label.setAlignment(QtCore.Qt.AlignCenter)
        self.heart_rate_monitor_label.setObjectName("heart_rate_monitor_label")
        self.horizontalLayout.addWidget(self.heart_rate_monitor_label)
        self.heart_rate_monitor_verticalLayout.addLayout(self.horizontalLayout)
        self.heart_rate_monitor_image_verticalLayout = QtWidgets.QVBoxLayout()
        self.heart_rate_monitor_image_verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.heart_rate_monitor_image_verticalLayout.setObjectName("heart_rate_monitor_image_verticalLayout")
        self.heart_rate_monitor_verticalLayout.addLayout(self.heart_rate_monitor_image_verticalLayout)
        self.heart_rate_monitor_verticalLayout.setStretch(1, 5)
        self.horizontalLayout_7.addLayout(self.heart_rate_monitor_verticalLayout)
        self.horizontalLayout_7.setStretch(1, 6)
        self.horizontalLayout_7.setStretch(2, 6)
        self.analysing_block_tabWidget.addTab(self.training_tab, "")
        self.analysing_tab = QtWidgets.QWidget()
        self.analysing_tab.setObjectName("analysing_tab")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.analysing_tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1151, 561))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.line_7 = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.verticalLayout_7.addWidget(self.line_7)
        self.train_plan_label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.train_plan_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.train_plan_label_2.setObjectName("train_plan_label_2")
        self.verticalLayout_7.addWidget(self.train_plan_label_2)
        self.video_resource_file_list = QtWidgets.QListWidget(self.horizontalLayoutWidget)
        self.video_resource_file_list.setObjectName("video_resource_file_list")
        self.verticalLayout_7.addWidget(self.video_resource_file_list)
        self.horizontalLayout_2.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.line_8 = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.verticalLayout_8.addWidget(self.line_8)
        self.train_plan_label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.train_plan_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.train_plan_label_3.setObjectName("train_plan_label_3")
        self.verticalLayout_8.addWidget(self.train_plan_label_3)
        self.history_train_result_info_widgetList = QtWidgets.QListWidget(self.horizontalLayoutWidget)
        self.history_train_result_info_widgetList.setObjectName("history_train_result_info_widgetList")
        self.verticalLayout_8.addWidget(self.history_train_result_info_widgetList)
        self.horizontalLayout_2.addLayout(self.verticalLayout_8)
        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 4)
        self.analysing_block_tabWidget.addTab(self.analysing_tab, "")
        self.rank_listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.rank_listWidget.setGeometry(QtCore.QRect(1210, 70, 231, 531))
        self.rank_listWidget.setObjectName("rank_listWidget")
        self.train_plan_label_4 = QtWidgets.QLabel(self.centralwidget)
        self.train_plan_label_4.setGeometry(QtCore.QRect(1210, 40, 221, 21))
        self.train_plan_label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.train_plan_label_4.setObjectName("train_plan_label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1455, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.analysing_block_tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.train_plan_label.setText(_translate("MainWindow", "训练方案："))
        self.current_datetime_label.setText(_translate("MainWindow", "当前时间："))
        self.current_datetime_value_label.setText(_translate("MainWindow", "2022-22-22 22:22:22"))
        self.count_down_label.setText(_translate("MainWindow", "倒计时："))
        self.count_down_value_label.setText(_translate("MainWindow", "10"))
        self.finish_percentage_label.setText(_translate("MainWindow", "完成进度："))
        self.finish_percentage_value_label.setText(_translate("MainWindow", "2 / 10"))
        self.train_display_area_label.setText(_translate("MainWindow", "训练画面"))
        self.monitor_hear_rate_comboBox.setText(_translate("MainWindow", "监测心率"))
        self.train_display_area_realtime_button.setText(_translate("MainWindow", "从摄像头播放"))
        self.train_display_area_play_video_button.setText(_translate("MainWindow", "选择视频播放"))
        self.train_display_area_stop_video_button.setText(_translate("MainWindow", "关闭画面"))
        self.heart_rate_monitor_label.setText(_translate("MainWindow", "心率监测"))
        self.analysing_block_tabWidget.setTabText(self.analysing_block_tabWidget.indexOf(self.training_tab), _translate("MainWindow", "训练"))
        self.train_plan_label_2.setText(_translate("MainWindow", "训练历史"))
        self.train_plan_label_3.setText(_translate("MainWindow", "单次训练数据"))
        self.analysing_block_tabWidget.setTabText(self.analysing_block_tabWidget.indexOf(self.analysing_tab), _translate("MainWindow", "分析"))
        self.train_plan_label_4.setText(_translate("MainWindow", "小程序答题耗时排行榜"))