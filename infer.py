#! /usr/bin/env python3
import warnings
import os, time
from mvnc import mvncapi as mvnc
import numpy
import cv2
import os, sys

# Load graph
path_to_networks = './Inception-v3/'
#path_to_images = dir
graph_filename = 'graph'
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

# Load categories
categories = []
with open(path_to_networks + 'categories.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    #print('Number of categories:', len(categories))

# Load dict
dict = []
with open(path_to_networks + 'dict.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        dict.append(cat)
    f.close()
    #print('Number of categories:', len(dict))

#Load inputsize
with open(path_to_networks + 'inputsize.txt', 'r') as f:
    reqsize = int(f.readline().split('\n')[0])

# Load preprocessing data
mean = 128
std = 1 / 128

#load Devices
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()
device = mvnc.Device(devices[0])
device.OpenDevice()
graph = device.AllocateGraph(graphfile)

cap = cv2.VideoCapture(0)
def capture():
    ret, frame = cap.read()
    return frame

def ImageRead():
    return cv2.imread("./test3.jpg")

def inference(subimages, number):
    for k in range(number):
        img = numpy.array(subimages).astype(numpy.float32)
        dx, dy, dz = img.shape
        delta = float(abs(dy - dx))
        if dx > dy:  # crop the x dimension
            img = img[int(0.5 * delta):dx - int(0.5 * delta), 0:dy]
        else:
            img = img[0:dx, int(0.5 * delta):dy - int(0.5 * delta)]
        img = cv2.resize(img, (reqsize, reqsize))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(3):
            img[:, :, i] = (img[:, :, i] - mean) * std

        #print('Start download to NCS...')
        graph.LoadTensor(img.astype(numpy.float16), 'user object')
        output, userobj = graph.GetResult()

        top_inds = output.argsort()[::-1][:5]


        #print(''.join(['*' for i in range(79)]))
        for i in range(5):
            if output[top_inds[i]] <= 0.001 or categories[top_inds[i]] not in dict:
                break
            # print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])
            print(categories[top_inds[i]])
        #print(''.join(['*' for i in range(79)]))



image = ImageRead()
#image = capture()
inference(image, 1)
