# -*- coding: utf-8 -*-
# @File : test.py 
# @Description : 
# @Author : Xiang Wentao, Software College of NEU
# @Contact : neu_xiangwentao@163.com
# @Date : 2023/5/5 16:39
import cv2


def videocapture():
    cap = cv2.VideoCapture(0)  # 生成读取摄像头对象
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
    # 定义视频对象输出
    writer = cv2.VideoWriter("video_result.mp4", fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()  # 读取摄像头画面
        cv2.imshow('video', frame)  # 显示画面
        key = cv2.waitKey(24)
        writer.write(frame)  # 视频保存
        # 按Q退出
        if key == ord('q'):
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 释放所有显示图像窗口

def sc():
    import requests
    url = "https://chat.aidutu.cn/api/cg/chatgpt/user/info?v=1.5"
    hs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.68",
        "Content-Type": "application/json",
        "Host": "chat.aidutu.cn",
        "Origin": "https://chat.aidutu.cn",
        "Referer": "https://chat.aidutu.cn/",
        "Accept": "application/json",
        "x-iam": "Dvpddoab",
        "x-version": "1.5",
    }
    dt = {
        "iam": "Dvpddoab",
        "isVip": 0,
        "q": "tell a joke",
    }
    res = requests.post(url, headers=hs, data=dt)
    print(res.status_code)
    import urllib3
    import urllib.parse
    import json
    print(urllib.parse.unquote(json.loads(res.text)['error_des']))

if __name__ == '__main__':
    # videocapture()
    sc()