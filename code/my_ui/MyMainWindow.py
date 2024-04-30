# -*- coding: utf-8 -*-
import json
import math
import multiprocessing
import os
import random
import shutil
import threading
import time

import requests
import torch
from PIL import Image
from matplotlib.pyplot import imshow
from torch.autograd import Variable

from models import LinkNet34
from my_mediapipe.MyMediaPipe import MyMediaPipe
from my_ui.HistoryGradeListItem import HistoryGradeListItem
from my_ui.HistoryVideoListItem import HistoryVideoListItem
from my_ui.RankListItem import RankListItem
from process_mask import ProcessMasks
from PyQt5.QtCore import QTimer, QDateTime, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QAbstractItemView
import torchvision.transforms as transforms

from os.path import isfile as os_path_isfile, exists as os_path_exists
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

from plot_cont import MyMatplotlibFigure, MyDynamicPlot
# from plot_cont import DynamicPlot
from ui.MainWindow import Ui_MainWindow
import cv2

from my_ui.MyTrainResult import MyTrainResult_Ui_Form as mtruf
from my_ui.my_video_play_window import My_Ui_MainWindow as Video_Play_Window

class My_Ui_MainWindow(QMainWindow, Ui_MainWindow):
    train_result_signal = pyqtSignal(dict)
    clear_camera_label_signal = pyqtSignal()
    add_one_history_grade_record_signal = pyqtSignal(dict)

    def __init__(self, sz=270, fs=28, bs=30, plot=False):
        super().__init__()
        self.setupUi(self)
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot
        self.is_action_begin = False
        self.pick_frame_interval = 4
        self.current_frame_count = 0
        self.current_finish_count = 0
        self.target_finish_count = 0
        self.video_record_history_folder_path = "./video_record_history"
        if os.path.exists(self.video_record_history_folder_path) is False:
            os.makedirs(self.video_record_history_folder_path)
        self.tmp_video_record_history_frames_folder_path = "./tmp"
        self.tmp_current_video_record_history_frames_folder_path = ""
        self.url_root = 'http://127.0.0.1:5000'
        self.url_name_submit_train_result = "submit_train_result"
        self.url_path_submit_train_result = "{}/{}".format(self.url_root, self.url_name_submit_train_result)



        self.frame_memory = []

        self.my_mediapipe = MyMediaPipe()

        self.frame_counter = 0
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth'))
        self.model.eval()
        self.model.to(self.device)

        self.init_multiprocessing_variables()
        self.videoCapture = None
        self.videoCapture = cv2.VideoCapture(0)
        def t(vc):
            try:
                _, a = vc.read()
                for _ in range(100):
                    self.mp_queue_for_hr_analysis.put(a)
            except:
                pass
        threading.Thread(target=t, args=[self.videoCapture]).start()

        self.train_plan_list = None
        self.mask_process_pipe, mask_process_receiver_pipe = multiprocessing.Pipe()
        data_from_MyDynamicPlot_sender_pipe, self.data_from_MyDynamicPlot_receiver_pipe = multiprocessing.Pipe()
        self.process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)
        self.plot_heart_rate_sender_pipe = None

        # plot_heart_rate_receiver_pipe是MyDynamicPlot类中负责接收心率数据的，plot_pipe是发送心率数据的
        # plot_heart_rate_receiver_pipe需要从MyDynamicPlot类接收到数据后再通过pipe发送到MyMatplotlibFigure类中，由MyMatplotlibFigure类进行绘制
        self.plot_heart_rate_sender_pipe, plot_heart_rate_receiver_pipe = multiprocessing.Pipe()
        self.plotter = MyDynamicPlot(self.signal_size,
                                     '0',
                                     self.batch_size,
                                     30,
                                     None
                                     )
        self.plot_process = multiprocessing.Process(target=self.plotter, args=(plot_heart_rate_receiver_pipe, data_from_MyDynamicPlot_sender_pipe), daemon=True)
        self.plot_process.start()

        self.mask_process = multiprocessing.Process(
            target=self.process_mask, args=(mask_process_receiver_pipe,
                                            self.plot_heart_rate_sender_pipe,
                                            self.get_current_frame_source()),
            daemon=True)
        self.mask_process.start()

        self.load_config()


        self.init_signals()
        self.init_events()
        self.init_timers()
        self.init_windows()
        self.init_widgets()
        self.init_processes()
        self.init_threads()
        self.setFixedSize(1455, 665)

        # self.mp_queue_consumer_process.join()

    def get_current_frame_source(self):
        if self.is_current_frame_from_camera_or_local_file() == 0:
            # 0-from file
            # 1-camera
            return "" # aaa
        else:
            return 0

    def init_threads(self):
        threading.Thread(target=self.mp_queue_consumer_for_hr_analysis).start()
        # threading.Thread(target=self.mp_queue_consumer_for_save_camera_frame_to_local).start()

    def init_processes(self):
        # self.mp_queue_consumer_process = multiprocessing.Process(target=self.mp_queue_consumer_for_hr_analysis,)
        # self.mp_queue_consumer_process.start()
        pass

    def init_multiprocessing_variables(self):
        self.mp_queue_for_hr_analysis = multiprocessing.Queue()
        # self.mp_queue_for_save_camera_frame_to_local = multiprocessing.Queue()

    def init_signals(self):
        self.clear_camera_label_signal.connect(self.clear_video_area)
        self.add_one_history_grade_record_signal.connect(self.add_one_history_grade_record_signal_callback)

    def init_windows(self):
        self.train_result_window = mtruf(self.train_result_signal, self.clear_camera_label_signal)

    def init_timers(self):
        print("init_debug_timer")
        self.target_finish_count = 194
        # 刷新倒计时的
        self.current_count_down_value = 0
        self.timer_count_down = QTimer()
        self.timer_count_down.timeout.connect(self.count_down_timer_timeout_callback)

        self.switch_countdown_and_finishcountprogress_timer("stop")

        # 刷新摄像头画面的
        self.timer_camera = QTimer()
        self.timer_camera.timeout.connect(self.timer_camera_timeout_callback)

        # 刷新当前时间的
        self.timer_update_current_datetime = QTimer()
        self.timer_update_current_datetime.timeout.connect(self.timer_update_current_datetime_timeout_callback)
        self.start_current_datetime_timer()

    def start_current_datetime_timer(self):
        self.timer_update_current_datetime.start(1000)

    def init_widgets(self):
        self.video_play_window = Video_Play_Window()
        self.update_train_plan_comboBox()

        self.canvas = MyMatplotlibFigure(self.data_from_MyDynamicPlot_receiver_pipe)
        # self.plotcos()
        threading.Thread(target=self.canvas.pipe_data_receiver_function).start()
        # self.timer_update_figure()
        self.heart_rate_monitor_image_verticalLayout.addWidget(self.canvas)
        self.init_countdown_and_finishcountlabel_and_progressbar()

        # 更新排行榜
        self.init_rank_list()

    def init_rank_list(self):
        res = requests.get("{}/get_fastest_5_user".format(self.url_root))
        print(res.text)
        ret = json.loads(res.text)
        fastest_user_info_list = ret['data']
        fastest_user_info_list = sorted(fastest_user_info_list, key=lambda x: x['finish_seconds'], reverse=True)
        # 清空排行榜现有的内容
        self.rank_listWidget.clear()
        for item in fastest_user_info_list:
            self.add_one_rank_list_item(item)

    def add_one_rank_list_item(self, data):
        RankListItem(self.rank_listWidget, data).add_item()

    def init_countdown_and_finishcountlabel_and_progressbar(self):
        self.count_down_timer_timeout_callback()
        self.update_progressBar(self.current_finish_count)
        self.update_finish_percentage_value(self.current_finish_count)

    def timer_update_figure(self):
        self.t = QTimer()
        # self.t.timeout.connect(self.canvas.pipe_data_receiver_function)
        self.t.timeout.connect(self.plotcos)
        self.t.start(100)

    def plotcos(self):
        t = np.arange(0.0, 5.0, 0.01)
        ri = random.randint(1, 5)
        s = np.cos(ri * np.pi * t)
        self.canvas.mat_plot_draw_axes(t, s)
        self.canvas.figs.suptitle("sin_{}".format(ri))  # 设置标题

    def filter_train_plan_names(self) -> list:
        name_list = []
        if self.train_plan_list is not None:
            for train_plan in self.train_plan_list:
                if 'name' in train_plan:
                    name_list.append(train_plan['name'])
        return name_list

    def update_train_plan_comboBox(self):
        name_list = self.filter_train_plan_names()
        name_list.insert(0, "请选择")
        if name_list is not None and len(name_list) > 0:
            self.add_item_list_to_train_plan_comboBox(name_list)

    def find_chosen_train_plan(self, train_plan_name):
        for item in self.train_plan_list:
            if 'name' in item:
                if item['name'] == train_plan_name:
                    return item
        return None

    def add_one_item_to_train_plan_comboBox(self, train_plan_item):
        self.train_plan_comboBox.addItem(train_plan_item)

    def add_item_list_to_train_plan_comboBox(self, train_plan_item_list):
        self.train_plan_comboBox.addItems(train_plan_item_list)
        pass

    def clear_train_plan_comboBox(self):
        self.train_plan_comboBox.clear()

    def closeEvent(self, e):
        # print(">> closeEvent")
        # self.plot_process.stop()
        # print(">> ")
        # self.mask_process.stop()
        # print(">> closeEvent")
        # self.videoCapture.release()
        # print(">> closeEvent")
        # self.videoCapture.close()
        # print(">> closeEvent")
        e.accept()

    def load_config(self):
        from config import configuration as cfg
        self.cfg = cfg
        self.train_plan_list = self.cfg['train_plan_list']

    def count_down_timer_timeout_callback(self):
        # print(">> count_down_timer_timeout_callback")
        self.count_down_value_label.setText("{}".format(self.current_count_down_value))
        self.update_count_down_value()

    def update_count_down_value(self):
        # print(">> update_count_down_value")
        if self.current_count_down_value > 0:
            self.current_count_down_value -= 1
        else:
            self.current_count_down_value = 0
        # print("> ", self.current_count_down_value)

    def judge_is_action_finished(self, chosen_train_plan_name):
        if chosen_train_plan_name == '肱三头肌训练':
            if self.is_action_begin is True:
                # 已经开始了，只需要判断是否结束

                self.is_action_begin = False
            else:
                # 还没开始，需要判断是否开始

                self.is_action_begin = True
            pass
        elif chosen_train_plan_name == '肱二头肌训练':
            if self.my_mediapipe.is_right_elbow_higher_than_right_wrist() is True:
                # print("elbow high")
                if self.is_action_begin == False:
                    self.is_action_begin = True
            elif self.my_mediapipe.is_right_elbow_lower_than_right_wrist() is True:
                # print("elbow low")
                if self.is_action_begin == True:
                    self.is_action_begin = False
                    return True
            else:
                print("invalid")
        elif chosen_train_plan_name == '三角肌前束训练':

            pass
        return False

    def is_training_finished(self):
        # print("self.current_finish_count\t", self.current_finish_count)
        # print("self.target_finish_count\t", self.target_finish_count)
        return (self.current_finish_count != 0 \
                and self.target_finish_count != 0 \
                and self.current_finish_count >= self.target_finish_count) \
               or self.current_count_down_value == 0

    def show_prompt(self, prompt_type):
        if prompt_type == 'training_finished':
            # 训练结束后弹窗提示查看训练结果
            box = QMessageBox(QMessageBox.Question,
                              "通知", "恭喜你！训练完成啦！",
                              QMessageBox.NoButton, self)
            box.addButton("查看训练结果！", QMessageBox.AcceptRole)
            reply = box.exec_()
            return reply
        elif prompt_type == 'click_to_start_training':
            # 选择完训练方案后弹窗提示是否开始训练
            # 训练结束后弹窗提示查看训练结果
            box = QMessageBox(QMessageBox.Question,
                              "通知", "已选训练方案[{}]，点击以开始训练！".format(
                    self.train_plan_comboBox.currentText()),
                              QMessageBox.NoButton, self)
            box.addButton("开始训练！", QMessageBox.AcceptRole)
            reply = box.exec_()
            return reply
        else:
            pass

        return None

    def update_finish_percentage_value(self, current_count):
        self.finish_percentage_value_label.setText("{}/{}".format(current_count, self.target_finish_count))

    def update_progressBar(self, current_count):
        self.progressBar.setValue(
            100.0 * current_count / self.target_finish_count if self.target_finish_count != 0 else 0
        )

    def start_capture_frame_timer(self, call_interval: int):
        # 更新全局的writer，用于将frame保存到不同的视频文件里
        # aaa
        # width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
        # height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
        # fps = self.videoCapture.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
        # fourcc = int(self.videoCapture.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
        # video_save_path = "{}/{}_{}.mp4".format(
        #         self.video_record_history_folder_path,
        #         self.current_datetime_value_label.text(),
        #         self.train_plan_comboBox.currentText())
        # print("save_path: {}".format(video_save_path))
        # self.local_video_file_filewriter = cv2.VideoWriter(
        #     video_save_path, fourcc, fps, (width, height))

        self.current_video_record_filename_hashcode = "{}_{}".format(
            self.train_plan_comboBox.currentText(),
            self.current_datetime_value_label.text().replace(":", "_")
        ).__hash__()
        self.tmp_current_video_record_history_frames_folder_path = "{}/{}".format(
            self.tmp_video_record_history_frames_folder_path,
            self.current_video_record_filename_hashcode,
        )

        if os.path.exists(self.tmp_current_video_record_history_frames_folder_path) is False:
            os.makedirs(self.tmp_current_video_record_history_frames_folder_path)
        if not self.timer_camera.isActive():
            # 未活动。可以启动....
            print("start timer...")
            self.timer_camera.start(call_interval)

    def mp_queue_consumer_for_save_camera_frame_to_local(self):
        while True:
            try:
                if not self.mp_queue_for_save_camera_frame_to_local.empty():
                    frame = self.mp_queue_for_save_camera_frame_to_local.get()  # 从队列中取出帧
                    # aaa
                    self.local_video_file_filewriter.write(frame)
                else:
                    multiprocessing.sleep(0.01)
            except:
                pass
        print('consumer exited')

    def mp_queue_consumer_for_hr_analysis(self):
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        while True:
            try:
                if not self.mp_queue_for_hr_analysis.empty():
                    frame = self.mp_queue_for_hr_analysis.get()  # 从队列中取出帧
                    # 这里可以进行帧的处理
                    frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)
                    a = img_transform(Image.fromarray(frame))
                    a = a.unsqueeze(0)
                    imgs = Variable(a.to(dtype=torch.float, device=self.device))
                    pred = self.model(imgs)
                    '''
                        pred就是mask
                    '''
                    mask = pred.data.cpu().numpy()
                    mask = mask.squeeze()
                    # imshow(mask)
                    mask = mask > 0.8
                    frame[mask == 0] = 0
                    self.mask_process_pipe.send([frame])
                else:
                    multiprocessing.sleep(0.01)
            except:
                pass
        print('consumer exited')

    # def frame_queue_producer(self, queue):
    #     cap = cv2.VideoCapture(0)  # 打开摄像头
    #     while True:
    #         ret, frame = cap.read()  # 读取帧
    #         if not ret:  # 如果没有帧了就退出循环
    #             break
    #         queue.put(frame)  # 存入队列
    #     cap.release()
    #     print('producer exited')

    def mediapipe_inference(self, raw_frame):
        self.my_mediapipe.process_one_frame_and_parse_landmarks(raw_frame)
        if self.judge_is_action_finished(self.chosen_train_plan_name) is True:
            # 已经完成了一次
            self.current_finish_count = self.target_finish_count if self.current_finish_count > self.target_finish_count \
                else self.current_finish_count + 1
            self.update_progressBar(self.current_finish_count)
            self.update_finish_percentage_value(self.current_finish_count)
        if self.is_training_finished():
            self.show_prompt("training_finished")
            # 清空mediapipe里已有的绘制好的图数据
            self.my_mediapipe.clear_drawed_image()
            # 将数据传给子窗口
            # print("训练结束了.....")
            data_to_send = {
                'name': self.train_plan_comboBox.currentText(),
                'finish_count': self.finish_percentage_value_label.text(),
                'seconds_last': self.current_count_down_value,
                'finish_datetime': self.current_datetime_value_label.text(),
                'train_result_id': "{}".format(self.current_video_record_filename_hashcode)
            }
            self.train_result_signal.emit(
                data_to_send
            )
            # 发送到服务端
            requests.post(
                url=self.url_path_submit_train_result,
                data=data_to_send
            )
            # print(">> show")
            self.train_result_window.show()
            # 清空当前计数，并且停止训练
            self.switch_countdown_and_finishcountprogress_timer("stop")
            self.current_finish_count = 0
            self.target_finish_count = 0
            self.current_count_down_value = 0
            self.update_finish_percentage_value(self.current_finish_count)
            # 重置训练计划的下拉框
            self.train_plan_comboBox.setCurrentIndex(0)
            # 重置进度条
            self.progressBar.setValue(0)
            # 重置倒计时的label
            self.count_down_value_label.setText("-")
            # 清空画面
            self.train_display_area_stop_video_button_clicked_callback()
            # 将保存的临时视频帧合并成一个文件，并清空临时视频帧数据
            # self.merge_temp_frames_to_a_video_file_and_remove_all_tmp_frames()
        else:
            print("训练还没结束...")

    def update_camera_frame_to_label(self):
        if self.videoCapture is None:
            return
        if self.is_trainging() == False:
            return
        hasFrame, frame = self.videoCapture.read()
        if hasFrame:
            # print(self.mp_queue_for_save_camera_frame_to_local.empty())
            # self.mp_queue_for_save_camera_frame_to_local.put(frame)
            # self.local_video_file_filewriter.write(frame)
            # aaa
            savepath = "{}/{}.jpg".format(
                self.tmp_current_video_record_history_frames_folder_path,
                len(os.listdir(
                    self.tmp_current_video_record_history_frames_folder_path)
                ))
            cv2.imwrite(
                savepath, frame
            )
            t = None
            # Capture frame-by-frame
            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # mediapipe推理
            self.current_frame_count = (self.current_frame_count + 1) % self.pick_frame_interval
            if self.is_training_finished() is False and self.current_frame_count == 0:
                print("mediapipe infer begin...")
                self.mediapipe_inference(frame)
                print("mediapipe infer end...")
                t = self.my_mediapipe.get_drawed_image()

            if 1 == 1 and\
                self.monitor_hear_rate_comboBox.isChecked():
                # 给心率模型推理：
                self.mp_queue_for_hr_analysis.put(frame)
            # 展示到画面区域
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = t if t is not None else frame
            frame_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            # print("set to pixmap>>>")
            self.train_display_area.setPixmap(QPixmap.fromImage(frame_image))
            # self.train_display_area.setScaledContents(True)


    def timer_update_current_datetime_timeout_callback(self):
        time_to_display_string = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.current_datetime_value_label.setText(time_to_display_string)

    def timer_camera_timeout_callback(self):
        try:
            self.update_camera_frame_to_label()
        except Exception:
            print("update frame to label failed.....")
            print(Exception.with_traceback())

    def init_events(self):
        # 播放本地文件
        self.train_display_area_play_video_button.clicked.connect(
            self.train_display_area_play_video_button_clicked_callback)
        # 实时
        self.train_display_area_realtime_button.clicked.connect(
            self.train_display_area_realtime_button_clicked_callback)
        # 停止播放实时画面or本地文件
        self.train_display_area_stop_video_button.clicked.connect(
            self.train_display_area_stop_video_button_clicked_callback)

        # 监听训练计划的改变
        self.train_plan_comboBox.currentIndexChanged.connect(self.train_plan_comboBox_currentIndexChanged_callback)
        # 监听tabWidget的切换
        self.analysing_block_tabWidget.currentChanged.connect(self.analysing_block_tabWidget_currentChanged)
        self.video_resource_file_list.itemClicked.connect(self.video_resource_file_list_itemClicked_event)

    def video_resource_file_list_itemClicked_event(self, item):
        print("======================")
        print(type(item))
        selected_item_path = item.get_video_file_path()
        selected_train_result_id = os.path.basename(selected_item_path)
        selected_train_result_id = selected_train_result_id[: selected_train_result_id.index(".mp4")]
        # self.init_all_history_grade_listWidget()
        self.set_selected_train_result_history_grade(selected_train_result_id)

    def set_selected_train_result_history_grade(self, selected_train_result_id):
        print('============= set_selected_train_result_history_grade =============')
        self.clear_history_train_result_info_widgetList()
        try:
            r = requests.get(url='http://127.0.0.1:5000/get_one_train_result_by_id?train_result_id={}'.format(selected_train_result_id))
            import urllib.parse
            import json
            rt = r.text
            rt_obj = json.loads(rt)
            history_grade_list = rt_obj['data']
            print(len(history_grade_list))
            for hg in history_grade_list:
                self.add_one_history_grade_record_signal.emit(hg)
        except Exception as e:
            print(e)
            pass
        pass

    def analysing_block_tabWidget_currentChanged(self):
        # if self.tabWidget.currentIndex()==0
        # 更新右侧的排行榜
        self.init_rank_list()
        if self.analysing_block_tabWidget.currentIndex() == 0:
            pass
        else:
            # 进入分析列表
            print('进入分析列表...')
            # 清空原本已有的
            self.video_resource_file_list.clear()
            self.init_video_listWidget()

    def video_resource_file_list_double_clicked_callback(self, modelindex: QModelIndex)->None:
        # print(modelindex)
        print(self.video_resource_file_list.currentItem())
        item = self.video_resource_file_list.currentItem()
        # print(type(item))
        print(item.get_video_file_path())
        print(self.video_resource_file_list.selectedItems())
        self.video_play_window.set_video_path(item.get_video_file_path())
        self.video_play_window.set_window_title(item.get_video_file_path())
        self.video_play_window.start_to_play()

    def get_uploaded_video_info_dict_list(self):
        all_video_file_path_list = ["{}/{}".format(self.video_record_history_folder_path, item) for item in os.listdir(self.video_record_history_folder_path)]
        '''
        ./video_record_history/aaa2.mp4
        '''
        ret = [
            {
                'file_path': p,
                'username': os.path.basename(p),
                'datetime': os.path.basename(p)[:os.path.basename(p).index(".mp4")]
            } for p in all_video_file_path_list]
        # data['file_path']
        # data['username']
        # data['datetime']
        return ret

    def init_video_listWidget(self):
        self.video_resource_file_list.doubleClicked.connect(self.video_resource_file_list_double_clicked_callback)
        # self.video_resource_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection) # 设置按住ctrl后可以多选
        uploaded_video_file_info_dict_list = self.get_uploaded_video_info_dict_list()
        # 渲染到listwidget里
        for item in uploaded_video_file_info_dict_list:
            self.add_one_video_signal_callback(item)

    def add_one_video_signal_callback(self, data: dict):
        print('============= add_one_video_signal_callback =============')
        HistoryVideoListItem(self.video_resource_file_list, data).add_item()

    def init_all_history_grade_listWidget(self):
        print('============= init_all_history_grade_listWidget =============')
        self.clear_history_train_result_info_widgetList()
        try:
            r = requests.get(url='http://127.0.0.1:5000/get_all_train_result')
            import urllib.parse
            import json
            rt = r.text
            rt_obj = json.loads(rt)
            history_grade_list = rt_obj['data']
            print(len(history_grade_list))
            for hg in history_grade_list:
                self.add_one_history_grade_record_signal.emit(hg)
        except Exception as e:
            print(e)
            pass

    def clear_history_train_result_info_widgetList(self):
        self.history_train_result_info_widgetList.clear()

    def add_one_history_grade_record_signal_callback(self, data:dict):
        print('============= add_one_history_grade_record_signal_callback =============')
        # {'train_plan_name': '肱二头肌训练', 'finish_count': '4/4', 'seconds_last': '52',
        #  'finish_datetime': '2023-05-08 14:17:47'}
        HistoryGradeListItem(self.history_train_result_info_widgetList, data).add_item()

    def train_plan_comboBox_currentIndexChanged_callback(self):
        self.is_action_begin = False
        chosen_train_plan_name = self.train_plan_comboBox.currentText()
        chosen_train_plan_info = self.find_chosen_train_plan(chosen_train_plan_name)
        self.current_finish_count = 0
        if chosen_train_plan_info is None:
            self.current_count_down_value = 0
            self.target_finish_count = 0
            # 关闭计时器
            self.switch_countdown_and_finishcountprogress_timer("stop")
        else:
            to_train = self.show_prompt("click_to_start_training")
            # print("start training.......")
            # print(chosen_train_plan_info['count_down'])
            self.chosen_train_plan_name = chosen_train_plan_name
            self.current_count_down_value = chosen_train_plan_info['count_down']
            # print(chosen_train_plan_info['target_finish_count'])
            self.target_finish_count = chosen_train_plan_info['target_finish_count']
            # 启动计时器
            self.switch_countdown_and_finishcountprogress_timer("start")
            # 开始播放画面——可能是从摄像头也可能是从本地文件，取决于用户是否选择了文件
            self.start_capture_frame_timer(np.round(1000/self.videoCapture.get(cv2.CAP_PROP_FPS)))
            # self.train_display_area_realtime_button_clicked_callback()

    def switch_countdown_and_finishcountprogress_timer(self, t):
        if t == "start":
            print("start timer...")
            self.timer_count_down.start(1000)
        elif t == 'stop':
            self.timer_count_down.stop()

    def is_current_frame_from_camera_or_local_file(self):
        '''
        :return: 0-from local file; 1-from camera
        '''
        if self.videoCapture.get(cv2.CAP_PROP_POS_AVI_RATIO) > 0:
            print("from video file")
            self.videoCapture = cv2.VideoCapture(0)
            self.process_mask.update_source(0)
            return 0
        else:
            print("from camera")
            return 1

    def train_display_area_play_video_button_clicked_callback(self):
        print("choose video file...")
        chosen_file_path = QFileDialog.getOpenFileUrl(self, "选择视频文件")
        # chosen_file_path = QFileDialog.getOpenFileUrl()
        chosen_file_path = chosen_file_path[0].url()
        chosen_file_path = chosen_file_path[8:]
        print('file url parsed...')
        print("video file path is [{}]...".format(chosen_file_path))
        if chosen_file_path == '':
            print('is empty')
            return
        if not os_path_exists(chosen_file_path):
            print("is not exist")
            return
        if not os_path_isfile(chosen_file_path):
            print("is not file")
            return
        print("video file path is valid. [{}] ...".format(chosen_file_path))
        self.videoCapture = cv2.VideoCapture(chosen_file_path)

    def train_display_area_stop_video_button_clicked_callback(self):
        if self.timer_camera.isActive():
            # 是正在播放摄像头视频
            print("stop timer...")
            self.timer_camera.stop()
        else:
            # 是正在播放本地的视频
            pass
        self.clear_video_area()
        # self.videoCapture = None
        # if self.is_trainging():
        self.switch_countdown_and_finishcountprogress_timer('stop')
        # 重置训练方案下拉框
        self.train_plan_comboBox.setCurrentIndex(0)
        # 将保存的临时视频帧合并成一个文件，并清空临时视频帧数据
        self.merge_temp_frames_to_a_video_file_and_remove_all_tmp_frames()

    def merge_temp_frames_to_a_video_file_and_remove_all_tmp_frames(self):
        if os.path.exists(self.tmp_current_video_record_history_frames_folder_path)\
                and len(os.listdir(self.tmp_current_video_record_history_frames_folder_path)) > 0:
            # 存在文件夹并且有保存帧
            width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
            height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
            fps = self.videoCapture.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
            fourcc = int(self.videoCapture.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
            video_save_path = "{}/{}.mp4".format(
                self.video_record_history_folder_path,
                os.path.basename(self.tmp_current_video_record_history_frames_folder_path)
            )
            print("save_path: {}".format(video_save_path))

            local_video_file_filewriter = cv2.VideoWriter(
                video_save_path, fourcc, fps, (width, height))
            for filename in sorted(os.listdir(self.tmp_current_video_record_history_frames_folder_path)):
                img = cv2.imread('{}/{}'.format(
                    self.tmp_current_video_record_history_frames_folder_path, filename))
                # resize方法是cv2库提供的更改像素大小的方法
                # 将图片转换为1280*720像素大小
                img = cv2.resize(img, (width, height))
                # 写入视频
                local_video_file_filewriter.write(img)

            print("删除：{}".format(self.tmp_current_video_record_history_frames_folder_path))
            shutil.rmtree(self.tmp_current_video_record_history_frames_folder_path)

    def train_display_area_realtime_button_clicked_callback(self):
        if self.videoCapture is None:
            # 此时赋值是安全的
            print("self.videoCapture is None")
            self.videoCapture = cv2.VideoCapture(0)
            self.process_mask.update_source(0)
        else:
            print("self.videoCapture is not None")
            if self.is_current_frame_from_camera_or_local_file() == 0:
                print("from video file")

                self.videoCapture = cv2.VideoCapture(0)
                self.process_mask.update_source(0)
            else:
                print("from camera")
            # if not self.videoCapture.isActive():
            #     print("self.videoCapture isActive")
            #     # 此时赋值是安全的
            #     self.videoCapture = cv2.VideoCapture(0)
            #     self.process_mask.update_source(0)
            # else:
            #     print("self.videoCapture is not Active")
            #     # 正在使用中.....不能赋值
            #     pass
        self.start_capture_frame_timer(np.round(1000/self.videoCapture.get(cv2.CAP_PROP_FPS)))

    def is_trainging(self):
        return self.timer_count_down.isActive() and self.timer_camera.isActive()

    def clear_video_area(self):
        self.train_display_area.clear()
