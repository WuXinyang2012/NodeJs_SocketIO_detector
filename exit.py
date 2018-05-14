#! /usr/bin/env python3
import warnings
import os, time
from mvnc import mvncapi as mvnc
import numpy
import cv2
import os, sys


#load Devices
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()
device = mvnc.Device(devices[0])
device.OpenDevice()
device.CloseDevice()
print("Server ends now.")