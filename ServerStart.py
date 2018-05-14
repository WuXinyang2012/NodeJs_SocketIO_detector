#! /usr/bin/env python3
import warnings
import os, time
from mvnc import mvncapi as mvnc
import numpy
import cv2
import os, sys

warnings.simplefilter("ignore", DeprecationWarning)

#load Devices
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()
device = mvnc.Device(devices[0])
#device.CloseDevice()
device.OpenDevice()


#print(device)
print('Server starts now.')