# -*- coding:utf-8 -*-
from models import LinkNet34
from plot_cont import MyDynamicPlot
# from plot_cont import DynamicPlot
from process_mask import ProcessMasks

import os
import multiprocessing as mp
from optparse import OptionParser
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys

class CaptureFrames():
    frame_speed = 0

    def __init__(self, bs, source, save_signal_sender, show_mask=False):
        self.frame_counter = 0
        self.batch_size = bs
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth'))
        # self.model.load_state_dict(torch.load('linknet.pth', map_location='cpu'))
        self.model.eval()
        self.model.to(self.device)
        self.show_mask = show_mask
        self.camera = cv2.VideoCapture(source)
        self.save_signal_sender = save_signal_sender
        CaptureFrames.frame_speed = self.camera.get(cv2.CAP_PROP_FPS)  # 帧速率
        print("frame_speed:", CaptureFrames.frame_speed)

    def __call__(self, pipe, source):
        self.pipe = pipe
        self.capture_frames(source)

    def capture_frames(self, source):
        img_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # print(source)
        camera = self.camera
        time.sleep(1)
        self.model.eval()
        (grabbed, frame) = camera.read()

        time_1 = time.time()
        self.frames_count = 0
        while grabbed:
            (grabbed, orig) = camera.read()
            if not grabbed:
                continue

            shape = orig.shape[0:2]
            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(256,256), cv2.INTER_LINEAR )

            k = cv2.waitKey(1)
            if k != -1:
                self.terminate(camera)
                break

            a = img_transform(Image.fromarray(frame))
            a = a.unsqueeze(0)
            imgs = Variable(a.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)

            pred= torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            mask = pred.data.cpu().numpy()
            mask = mask.squeeze()

            # im = Image.fromarray(mask)
            # im2 = im.filter(ImageFilter.MinFilter(3))
            # im3 = im2.filter(ImageFilter.MaxFilter(5))
            # mask = np.array(im3)

            mask = mask > 0.8
            orig[mask==0]=0
            self.pipe.send([orig])

            if self.show_mask:
                cv2.imshow('mask', orig)

            if self.frames_count % 30 == 29:
                self.frames_count = -1
                time_2 = time.time()
                sys.stdout.write(f'\rFPS: {30/(time_2-time_1)}')
                sys.stdout.flush()
                time_1 = time.time()
                self.save_signal_sender.send("")

            self.frames_count+=1
        self.terminate(camera)

    def terminate(self, camera):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()

class RunPOS():
    def __init__(self,  sz=270, fs=28, bs=30, plot=False):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz
        self.plot = plot

    def __call__(self, source, fig, pulse_ax, hr_axis):
        time1=time.time()
        
        mask_process_pipe, chil_process_pipe = mp.Pipe()

        save_signal_recv, save_signal_sender = mp.Pipe()

        capture = CaptureFrames(self.batch_size, source, save_signal_sender,
                                show_mask=True)
        self.plot_pipe = None
        if self.plot:
            # plotter_pipe是接收心率数据的，plot_pipe是发送心率数据的
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = MyDynamicPlot(self.signal_size,
                                       os.path.basename(source) if (source != 0 and source != '0') \
                                            else '0',
                                       self.batch_size,
                                       CaptureFrames.frame_speed,
                                       save_signal_recv,
                                         fig, pulse_ax, hr_axis
            )
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        
        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)

        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, source, ), daemon=True)
        mask_processer.start()
        
        capture(mask_process_pipe, source)

        mask_processer.join()
        if self.plot:
            self.plot_process.join()
        time2=time.time()
        time2=time.time()
        print(f'time {time2-time1}')

def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--source', dest='source', default=0,
                        help='Signal Source: 0 for webcam or file path')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=30,
                        type='int', help='batch size')
    parser.add_option('-f', '--frame-rate', dest='framerate', default=25,
                        help='Frame Rate')
    (options, _) = parser.parse_args()
    return options

def run_POS(fig, pulse_ax, hr_axis):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    args = get_args()
    source = args.source
    # source = 'E:/test-01.mp4'
    runPOS = RunPOS(270, args.framerate, args.batchsize, False)
    runPOS(source, fig, pulse_ax, hr_axis)

if __name__=="__main__":
    run_POS()
