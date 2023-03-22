#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, request
import cv2
import os
import threading
import torch
import numpy as np
import copy
import random

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

global infer_buf
infer_buf = {
    "frame": None,
    "res": []
}

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
names =  {}

for k, v in model.names.items():
    names[k] = (v.replace(" ", "_"), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

def gen_frames():
    global infer_buf

    while True:
        success, frame = camera.read()
        if not success:
            print("ending...")
            break
        else:
            results = model(frame)

            cur_res = []
            for xmin, ymin, xmax, ymax, cof, cls_idx in results.xyxy[0]:
                cur_res.append([xmin.item(), ymin.item(), xmax.item(), ymax.item(), cof, int(cls_idx)])

            infer_buf = {
                "frame": frame,
                "res": cur_res
            }
            # results = np.squeeze(results.render())
            # result = frame
            # frame_buf = results
            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            # frame_buf = (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def render_bbox(frame, res, visible_set):
    for r in res:
        label, color = names[r[5]][0], names[r[5]][1]
        if ("all" not in visible_set) and (label not in visible_set): continue

        cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 3)
        cv2.putText(frame, label, (int(r[0]), int(r[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame

def read_frames(width, items, grey):
    resol_dict = {
        '360': [640, 360],
        '480': [854, 480],
        '720': [1280, 720],
        '1080': [1920, 1080]
    }
    resolution = (resol_dict[width][0], resol_dict[width][1])

    items = set(items)
    while True:
        cur_infer_buf = copy.deepcopy(infer_buf)
        frame = render_bbox(cur_infer_buf["frame"], cur_infer_buf["res"], items)
        resized_frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

        print(type(grey))
        if grey == 1:
            resized_frame=cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)   

        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        frame = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        yield frame

@app.route('/video_feed')
def video_feed():
    width = str(request.args.get('width', default=480, type=int))
    items = request.args.getlist("item")
    grey = request.args.get('grey', default=0, type=int)
    print(f"grey = {grey}")
    # item = str(request.args.get('item', default="nothing", type=str))
    print(width, items, threading.get_ident())
    
    return Response(read_frames(width, items, grey), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

t = threading.Thread(target=gen_frames)
t.start()

# if __name__ == '__main__':
#     app.run('0.0.0.0', port=3333, threaded=True)