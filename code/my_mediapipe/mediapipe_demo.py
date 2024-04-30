import cv2
import my_mediapipe as mp
import cv2
import numpy as np

import os

import win32api
import win32con
import win32gui
import win32ui

import time
from threading import Thread, Event

import keyboard

# class MyKeyBoardInput():
#     VK_CODE = {
#     'backspace': 0x08,
#     'tab': 0x09,
#     'clear': 0x0C,
#     'enter': 0x0D,
#     'shift': 0x10,
#     'ctrl': 0x11,
#     'alt': 0x12,
#     'pause': 0x13,
#     'caps_lock': 0x14,
#     'esc': 0x1B,
#     'spacebar': 0x20,
#     'page_up': 0x21,
#     'page_down': 0x22,
#     'end': 0x23,
#     'home': 0x24,
#     'left_arrow': 0x25,
#     'up_arrow': 0x26,
#     'right_arrow': 0x27,
#     'down_arrow': 0x28,
#     'select': 0x29,
#     'print': 0x2A,
#     'execute': 0x2B,
#     'print_screen': 0x2C,
#     'ins': 0x2D,
#     'del': 0x2E,
#     'help': 0x2F,
#     '0': 0x30,
#     '1': 0x31,
#     '2': 0x32,
#     '3': 0x33,
#     '4': 0x34,
#     '5': 0x35,
#     '6': 0x36,
#     '7': 0x37,
#     '8': 0x38,
#     '9': 0x39,
#     'a': 0x41,
#     'b': 0x42,
#     'c': 0x43,
#     'd': 0x44,
#     'e': 0x45,
#     'f': 0x46,
#     'g': 0x47,
#     'h': 0x48,
#     'i': 0x49,
#     'j': 0x4A,
#     'k': 0x4B,
#     'l': 0x4C,
#     'm': 0x4D,
#     'n': 0x4E,
#     'o': 0x4F,
#     'p': 0x50,
#     'q': 0x51,
#     'r': 0x52,
#     's': 0x53,
#     't': 0x54,
#     'u': 0x55,
#     'v': 0x56,
#     'w': 0x57,
#     'x': 0x58,
#     'y': 0x59,
#     'z': 0x5A,
#     'numpad_0': 0x60,
#     'numpad_1': 0x61,
#     'numpad_2': 0x62,
#     'numpad_3': 0x63,
#     'numpad_4': 0x64,
#     'numpad_5': 0x65,
#     'numpad_6': 0x66,
#     'numpad_7': 0x67,
#     'numpad_8': 0x68,
#     'numpad_9': 0x69,
#     'multiply_key': 0x6A,
#     'add_key': 0x6B,
#     'separator_key': 0x6C,
#     'subtract_key': 0x6D,
#     'decimal_key': 0x6E,
#     'divide_key': 0x6F,
#     'F1': 0x70,
#     'F2': 0x71,
#     'F3': 0x72,
#     'F4': 0x73,
#     'F5': 0x74,
#     'F6': 0x75,
#     'F7': 0x76,
#     'F8': 0x77,
#     'F9': 0x78,
#     'F10': 0x79,
#     'F11': 0x7A,
#     'F12': 0x7B,
#     'F13': 0x7C,
#     'F14': 0x7D,
#     'F15': 0x7E,
#     'F16': 0x7F,
#     'F17': 0x80,
#     'F18': 0x81,
#     'F19': 0x82,
#     'F20': 0x83,
#     'F21': 0x84,
#     'F22': 0x85,
#     'F23': 0x86,
#     'F24': 0x87,
#     'num_lock': 0x90,
#     'scroll_lock': 0x91,
#     'left_shift': 0xA0,
#     'right_shift ': 0xA1,
#     'left_control': 0xA2,
#     'right_control': 0xA3,
#     'left_menu': 0xA4,
#     'right_menu': 0xA5,
#     'browser_back': 0xA6,
#     'browser_forward': 0xA7,
#     'browser_refresh': 0xA8,
#     'browser_stop': 0xA9,
#     'browser_search': 0xAA,
#     'browser_favorites': 0xAB,
#     'browser_start_and_home': 0xAC,
#     'volume_mute': 0xAD,
#     'volume_Down': 0xAE,
#     'volume_up': 0xAF,
#     'next_track': 0xB0,
#     'previous_track': 0xB1,
#     'stop_media': 0xB2,
#     'play/pause_media': 0xB3,
#     'start_mail': 0xB4,
#     'select_media': 0xB5,
#     'start_application_1': 0xB6,
#     'start_application_2': 0xB7,
#     'attn_key': 0xF6,
#     'crsel_key': 0xF7,
#     'exsel_key': 0xF8,
#     'play_key': 0xFA,
#     'zoom_key': 0xFB,
#     'clear_key': 0xFE,
#     '+': 0xBB,
#     ',': 0xBC,
#     '-': 0xBD,
#     '.': 0xBE,
#     '/': 0xBF,
#     '`': 0xC0,
#     ';': 0xBA,
#     '[': 0xDB,
#     '\\': 0xDC,
#     ']': 0xDD,
#     "'": 0xDE,
#     '`': 0xC0
#     }
#
#     def input_char(self, char:str):
#         pass
#
#     def simulate_keyboard(self):
#         win32api.keybd_event(MyKeyBoardInput.VK_CODE['a'], 0, 0, 0)
#         win32api.keybd_event(MyKeyBoardInput.VK_CODE['a'], 0, win32con.KEYEVENTF_KEYUP, 0)
#         time.sleep(1)

class KeyBoardInputThread(Thread):
    def __init__(self): 
        super().__init__() 
        self.stop = False 
        self.print = Event()
        self.my_keyboard_input = MyKeyBoardInput()

    def run(self): 
        while not self.stop:
            if self.print.wait(1): 
                self.my_keyboard_input.simulate_keyboard()

    def join(self, timeout=None): 
        self.stop = True 
        super().join(timeout)

class MyMediaPipe():
    def __init__(self) -> None:
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.width = 500
        self.height = 300
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=True,
            smooth_landmarks=True,
            # model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    # def convert_landmark_to_screen_location(box_x, box_y, landmark_x, landmark_y):


    def monitor(self):
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            x,y = pyautogui.position()

            x = int(x - self.width/2)
            y = int(y - self.height/2)

            image = pyautogui.screenshot(region=[x, y, int(self.width), int(self.height)])  # 分别代表：左上角坐标，宽高
            #对获取的图片转换成二维矩阵形式，后再将RGB转成BGR
            #因为imshow,默认通道顺序是BGR，而pyautogui默认是RGB所以要转换一下，不然会有点问题
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            # image是以鼠标为中心的矩形

            # cv2.imshow("截屏", image)
            # cv2.waitKey(5)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # results = holistic.process(image)
            results = self.pose.process(image)
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                land_mark = results.pose_landmarks.landmark
                # 横着向右为x轴正向
                # 竖着向下为y轴正向

                # 获取头部所有关键点的坐标
                head_point_list = []
                for i in range(4):
                    # 下标为0~10的点为头部的点
                    head_point_list.append(land_mark[i])
                # print(len(head_point_list))
                visib_list = [_.visibility for _ in head_point_list]
                max_index = visib_list.index(max(visib_list))

                # print(max_index)
                # print(head_point_list[max_index])

                liable_point = head_point_list[max_index] # 这里应换成置信度最高的一个点
                nose_location_pix_x, nose_location_pix_y = int(liable_point.x * self.width), int(liable_point.y * self.height)
                # ↑ x 和 y 是在截图区域的图片中的像素点坐标，还需要转换为屏幕的坐标

                nose_location_pix_screen_x = x + nose_location_pix_x
                nose_location_pix_screen_y = y + nose_location_pix_y - 50
                print(x, y)
                print(liable_point.x, liable_point.y)
                print(nose_location_pix_x, nose_location_pix_y)
                print(nose_location_pix_screen_x, nose_location_pix_screen_y)
                print()
                # 移动鼠标到这个点
                pyautogui.moveTo(nose_location_pix_screen_x, nose_location_pix_screen_y)

            # 再检测一次人体
            

            #画图
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.face_landmarks,
            #     mp_holistic.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_pose_landmarks_style())

            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.imshow('MediaPipe Holistic', image)
            cv2.waitKey(5)

class MouseMoveThread(Thread):
    def __init__(self): 
        super().__init__() 
        self.stop = False 
        self.mp_obj = MyMediaPipe()
        self.print = Event()

    def run(self): 
        while not self.stop:
            if self.print.wait(1): 
                self.mp_obj.monitor()
    
    def join(self, timeout=None): 
        self.stop = True 
        super().join(timeout)

from PIL import Image
from PIL import ImageChops
class MyFindDifference():
    def findTitle(self, window_title):
        '''
        查找指定标题窗口句柄
        @param window_title: 标题名
        @return: 窗口句柄
        '''
        hWndList = []
        # 函数功能：该函数枚举所有屏幕上的顶层窗口，办法是先将句柄传给每一个窗口，然后再传送给应用程序定义的回调函数。
        win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), hWndList)
        for hwnd in hWndList:
            # 函数功能：该函数获得指定窗口所属的类的类名。
            # clsname = win32gui.GetClassName(hwnd)
            # 函数功能：该函数将指定窗口的标题条文本（如果存在）拷贝到一个缓存区内
            title = win32gui.GetWindowText(hwnd)
            if (title == window_title):
                print("标题：", title, "句柄：", hwnd)
                break
        return hwnd

    def __init__(self):
# 391 35  # 窗口左上角的坐标x, y

# 435 423  # 第一张图的左上角的坐标x+44, y+388

# 1006 423  # 第二张图的左上角的坐标x+615, y+388

        self.width = 477
        self.height = 358
        self.area_1_x = 750
        self.area_2_x = self.area_1_x + 570
        self.area_1_y = 399
        self.area_2_y = 399
        self.area_1 = {}
        self.area_1['x'] = self.area_1_x
        self.area_1['y'] = self.area_1_y
        self.area_1['width'] = self.width
        self.area_1['height'] = self.height

        self.area_2 = {}
        self.area_2['x'] = self.area_2_x
        self.area_2['y'] = self.area_2_y
        self.area_2['width'] = self.width
        self.area_2['height'] = self.height


    def get_screenshot(self, area_1):
        image1 = pyautogui.screenshot(region=[
            int(area_1['x']), int(area_1['y']), int(area_1['width']), int(area_1['height'])
        ])  
        # 分别代表：左上角坐标，宽高
        #对获取的图片转换成二维矩阵形式，后再将RGB转成BGR

        #因为imshow,默认通道顺序是BGR，而pyautogui默认是RGB所以要转换一下，不然会有点问题
        image1 = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR)
        image1.flags.writeable = False
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        return image1

    def compare(self):
        self.compare_areas(self.area_1, self.area_2)

    def compare_areas(self, area_1, area_2):
        """
        比较图片，如果有不同则生成展示不同的图片
        @参数一: path_one: 第一张图片的路径
        @参数二: path_two: 第二张图片的路径
        @参数三: diff_save_location: 不同图的保存路径
        """
        image1 = self.get_screenshot(area_1)
        image1 = Image.fromarray(image1)

        image2 = self.get_screenshot(area_2)
        image2 = Image.fromarray(image2)

        try:
            diff = ImageChops.difference(image1, image2)
            if diff.getbbox() is not None:
                cv2.imshow('', np.array(diff))
                # diff.show()
                cv2.waitKey(0)
        except ValueError as e:
            text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
                "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
                "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
                "image must match the size of the region.使用2纬的box避免上述问题")
            print("【{0}】{1}".format(e,text))

    def compare_images(self, path_one, path_two, diff_save_location):
        """
        比较图片，如果有不同则生成展示不同的图片
        @参数一: path_one: 第一张图片的路径
        @参数二: path_two: 第二张图片的路径
        @参数三: diff_save_location: 不同图的保存路径
        """
        image_one = Image.open(path_one)
        image_two = Image.open(path_two)
        try:
            diff = ImageChops.difference(image_one, image_two)
            if diff.getbbox() is not None:
                diff.save(diff_save_location)
        except ValueError as e:
            text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
                "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
                "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
                "image must match the size of the region.使用2纬的box避免上述问题")
            print("【{0}】{1}".format(e,text))

class FindDifferenceThread(Thread):
    def __init__(self): 
        super().__init__() 
        self.stop = False 
        self.obj = MyFindDifference()
        self.print = Event()

    def run(self): 
        while not self.stop:
            if self.print.wait(1): 
                self.obj.compare()
    
    def join(self, timeout=None): 
        self.stop = True 
        super().join(timeout)

class KeyboardHook: 
    def __init__(self):
        self.printer = MouseMoveThread()
        self.printer = KeyBoardInputThread()
        self.printer = FindDifferenceThread()
        self.set_keyboard_hotkeys()

    def toggle_print(self): 
        print('Toggle the printer thread...')

        if self.printer.print.is_set():
            self.printer.print.clear()
        else:
            self.printer.print.set()

    def set_keyboard_hotkeys(self):
        print('Setting keyboard hotkeys...')
        keyboard.add_hotkey('ctrl+alt+c', self.toggle_print)

    def start(self): 
        self.printer.start()

        try:
            keyboard.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.printer.join()

def simulate():
    hook = KeyboardHook()
    hook.start()

if '__main__' == __name__:
    simulate()
