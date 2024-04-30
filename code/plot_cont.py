import os
import sys

import numpy as np
import matplotlib
import pandas as pd

# matplotlib.use('TkAgg')
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import medfilt, decimate
import numpy as np
from matplotlib.figure import Figure

plt.ion()
class MyDynamicPlot():
    def __init__(self, signal_size, video_filename, bs, frame_speed, save_signal_recv):
        super().__init__()
        # self.fig, self.pulse_ax, self.hr_axis = fig, pulse_ax, hr_axis
        self.batch_size = bs
        self.signal_size = signal_size
        self.launched = False
        self.video_filename = video_filename
        # self.cnt = 0
        self.frame_cnt = 0
        self.frame_speed = frame_speed
        self.last_second = 0
        self.save_signal_recv = save_signal_recv
        self.heart_rate_to_save_to_excel_list = []
        self.pulse_to_plot = np.zeros(self.signal_size)
        self.hrs_to_plot = np.zeros(self.signal_size)

        video_result_root_path = "./result_data/{}".format(self.video_filename)
        if not os.path.exists("{}/images".format(video_result_root_path)):
            os.makedirs("{}/images".format(video_result_root_path))  # 存储图片的文件夹
        self.pulse_rate_image_save_folder_path = "{}/images/pulse".format(video_result_root_path)
        if not os.path.exists(self.pulse_rate_image_save_folder_path):
            os.makedirs(self.pulse_rate_image_save_folder_path)  # 存储图片的文件夹
        self.heart_rate_image_save_folder_path = "{}/images/heart_rate".format(video_result_root_path)
        if not os.path.exists(self.heart_rate_image_save_folder_path):
            os.makedirs(self.heart_rate_image_save_folder_path)  # 存储图片的文件夹
        heart_rate_data_in_excel_save_folder_path = "{}/hear_rate_data_in_excel".format(video_result_root_path)
        if not os.path.exists(heart_rate_data_in_excel_save_folder_path):
            os.makedirs(heart_rate_data_in_excel_save_folder_path)  # 存储实时心率的文件夹
        self.heart_rate_data_in_excel_save_file_path = heart_rate_data_in_excel_save_folder_path + '/{}.csv'.format(
            self.video_filename)

    def save_subfig(self, fig, ax, save_path, fig_name):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
        extent = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(save_path + '/' + fig_name, bbox_inches=extent)

    def launch_fig(self):
        self.pulse_to_plot = np.zeros(self.signal_size)
        self.hrs_to_plot = np.zeros(self.signal_size)
        # self.fig, (self.pulse_ax, self.hr_axis) = plt.subplots(2, 1)
        # self.pulse_to_plot = np.zeros(self.signal_size)
        # self.hrs_to_plot = np.zeros(self.signal_size)
        #
        # self.hr_texts = self.pulse_ax.text(0.1, 0.9, '0', ha='center', va='center', transform=self.pulse_ax.transAxes)
        # self.pulse_ax.set_title('BVP')
        # self.hr_axis.set_title('Heart Rate')
        # self.pulse_ax.set_autoscaley_on(True)
        #
        # self.pulse_ax.plot(self.pulse_to_plot)
        # self.hr_axis.plot(self.hrs_to_plot)
        #
        # self.pulse_ax.set_ylim(-3, 3)
        # self.hr_axis.set_ylim(0, 180)
        # # self.launched = True
        #
        # plt.tight_layout()
        # # plt.show()

    def __call__(self, pulse_hrs_data_receiver_pipe, data_from_MyDynamicPlot_sender_pipe):
        # if self.launched == False:
        #     self.launch_fig()
        # self.launch_fig()
        self.pulse_hrs_data_receiver_pipe = pulse_hrs_data_receiver_pipe
        self.data_from_MyDynamicPlot_sender_pipe = data_from_MyDynamicPlot_sender_pipe

        self.call_back()

    def call_back(self):
        while True:
            # d = self.save_signal_recv.recv()
            # if d is not None:
            #     self.last_second += 1
            #     self.save_each_plot()
            #     self.save_current_heart_rate()
            data = self.pulse_hrs_data_receiver_pipe.recv()
            # if data is not None:
            #     sys.stdout.write("len: {}".format(len(data)))
            #     sys.stdout.write('\n')
            #     # sys.stdout.write("data: {}".format(data))
            #     sys.stdout.flush()
            if data is None:
                self.terminate()
                break
            elif data == 'no face detected':
                self.update_no_face()
            else:
                self.update_data(data[0], data[1])

    def save_current_heart_rate(self):
        self.heart_rate_to_save_to_excel_list.append(self.hrs_to_plot[-1])

    def update_no_face(self):
        # hr_text = 'HR: NaN'
        # self.hr_texts.set_text(hr_text)

        scaled = np.zeros(10)
        for i in range(0, len(scaled)):
            self.pulse_to_plot[0:self.signal_size - 1] = self.pulse_to_plot[1:]
            self.pulse_to_plot[-1] = scaled[i]
            # self.update_plot(self.pulse_ax, self.pulse_to_plot)

            self.hrs_to_plot[0:self.signal_size - 1] = self.hrs_to_plot[1:]
            self.hrs_to_plot[-1] = 0
            # self.update_plot(self.hr_axis, self.hrs_to_plot)
            # self.re_draw()
            self.data_from_MyDynamicPlot_sender_pipe.send(
                self.construct_pulse_and_hrs_dict_data(self.pulse_to_plot, self.hrs_to_plot)
            )


    def construct_pulse_and_hrs_dict_data(self, pulse_to_plot, hrs_to_plot):
        return {
            "pulse_ax": {
                "x": np.arange(len(pulse_to_plot)),
                "y": pulse_to_plot
            },
            "hr_axis": {
                "x": np.arange(len(hrs_to_plot)),
                "y": hrs_to_plot
            },
        }

    def update_data(self, p, hrs):
        hr_fft = moving_avg(hrs, 3)[-1] if len(hrs) > 5 else hrs[-1]
        # hr_text = 'HR: ' + str(int(hr_fft))
        # self.hr_texts.set_text(hr_text)

        # ma = moving_avg(p[-self.batch_size:], 6)
        # sys.stdout.write("batch_size: {}".format(self.batch_size))
        # sys.stdout.flush()
        self.frame_cnt += self.batch_size
        batch = p[-self.batch_size:]
        decimated_p = decimate(batch, 3)
        # filterd_p =  medfilt(decimated_p, 5)
        scaled = scale_pulse(decimated_p)
        # self.cnt += 1
        for i in range(0, len(scaled)):
            self.pulse_to_plot[0:self.signal_size - 1] = self.pulse_to_plot[1:]
            self.pulse_to_plot[-1] = scaled[i]
            # self.update_plot(self.pulse_ax, self.pulse_to_plot)

            self.hrs_to_plot[0:self.signal_size - 1] = self.hrs_to_plot[1:]
            self.hrs_to_plot[-1] = hr_fft
            # self.update_plot(self.hr_axis, self.hrs_to_plot)
            # self.re_draw()

            self.data_from_MyDynamicPlot_sender_pipe.send(self.construct_pulse_and_hrs_dict_data(self.pulse_to_plot, self.hrs_to_plot))

    def update_plot(self, axis, y_values):
        line = axis.lines[0]
        line.set_xdata(np.arange(len(y_values)))
        line.set_ydata(y_values)
        axis.relim()
        axis.autoscale_view()

    def re_draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # def save_each_plot(self):
    #     self.save_subfig(self.fig, self.pulse_ax,
    #                      self.pulse_rate_image_save_folder_path,
    #                      "{}_{}.png".format(self.video_filename, self.last_second))
    #     self.save_subfig(self.fig, self.hr_axis,
    #                      self.heart_rate_image_save_folder_path,
    #                      "{}_{}.png".format(self.video_filename, self.last_second))

    def terminate(self):
        """
        saves numpy array of rPPG signal as pulse
        """
        np.save('pulse', self.pulse_to_plot)
        # print("------------------------------- close")
        df = pd.DataFrame()
        df['t'] = self.heart_rate_to_save_to_excel_list
        df = df.transpose()
        df.to_csv(self.heart_rate_data_in_excel_save_file_path, index=False, header=0)
        # plt.close('all')

class MyMatplotlibFigure(FigureCanvasQTAgg):
    """
    创建一个画布类，并把画布放到FigureCanvasQTAgg
    """
    def __init__(self, data_from_MyDynamicPlot_receiver_pipe, width=10, heigh=10, dpi=100):
        # plt.rcParams['figure.facecolor'] = 'r'  # 设置窗体颜色
        # plt.rcParams['axes.facecolor'] = 'b'  # 设置绘图区颜色
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        # fig, (pulse_ax, hr_axis) = plt.subplots(2, 1)
        # self.figs = fig
        self.figs = Figure(figsize=(width, heigh), dpi=dpi)
        super(MyMatplotlibFigure, self).__init__(self.figs)  # 在父类种激活self.fig，
        self.pulse_ax = self.figs.add_subplot(211)  # 添加绘图区
        self.hr_axis = self.figs.add_subplot(212)  # 添加绘图区
        self.figs.subplots_adjust(hspace=0.5)
        # self.fig, (self.pulse_ax, self.hr_axis) = plt.subplots(2, 1)
        self.data_from_MyDynamicPlot_receiver_pipe = data_from_MyDynamicPlot_receiver_pipe
        self.launch_fig()

    def launch_fig(self):
        # self.fig, (self.pulse_ax, self.hr_axis) = plt.subplots(2, 1)
        self.hr_texts = self.pulse_ax.text(0.1, 0.9, '0', ha='center', va='center', transform=self.pulse_ax.transAxes)
        self.pulse_ax.set_title('BVP')
        self.hr_axis.set_title('Heart Rate')
        self.pulse_ax.set_autoscaley_on(True)

        pulse_to_plot = np.zeros(270)
        hrs_to_plot = np.zeros(270)
        self.pulse_ax.plot(pulse_to_plot)
        self.hr_axis.plot(hrs_to_plot)

        self.pulse_ax.set_ylim(-3, 3)
        self.hr_axis.set_ylim(0, 180)

        # plt.tight_layout()

    def pipe_data_receiver_function(self):
        while True:
            pulse_and_hrs_dict_data = self.data_from_MyDynamicPlot_receiver_pipe.recv()
            if pulse_and_hrs_dict_data is None:
                continue
            self.mat_plot_draw_axes_pulse_ax(pulse_and_hrs_dict_data['pulse_ax']['x'], pulse_and_hrs_dict_data['pulse_ax']['y'])
            self.mat_plot_draw_axes_hrs_ax(pulse_and_hrs_dict_data['hr_axis']['x'], pulse_and_hrs_dict_data['hr_axis']['y'])
            self.redraw()

    def mat_plot_draw_axes_pulse_ax(self, t, s):
        """
        用清除画布刷新的方法绘图
        :return:
        """
        pass
        self.pulse_ax.cla()  # 清除绘图区

        self.pulse_ax.spines['top'].set_visible(False)  # 顶边界不可见
        self.pulse_ax.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        # self.pulse_ax.spines['bottom'].set_position(('data', 0))  # 设置y轴线原点数据为 0
        self.pulse_ax.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        # self.pulse_ax.plot(t, s, 'o-r', linewidth=0.5)
        self.pulse_ax.plot(t, s, linewidth=0.5)
        self.pulse_ax.set_title("BVP")

    def mat_plot_draw_axes_hrs_ax(self, t, s):
        """
        用清除画布刷新的方法绘图
        :return:
        """
        pass
        self.hr_axis.cla()  # 清除绘图区

        self.hr_axis.spines['top'].set_visible(False)  # 顶边界不可见
        self.hr_axis.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        # self.pulse_ax.spines['bottom'].set_position(('data', 0))  # 设置y轴线原点数据为 0
        self.hr_axis.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        # self.hr_axis.plot(t, s, 'o-r', linewidth=0.5)
        self.hr_axis.plot(t, s, linewidth=0.5)
        self.hr_axis.set_title("Heart Rate")

    def redraw(self):
        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas
        self.figs.subplots_adjust(hspace=0.5)

