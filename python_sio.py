#!/usr/bin/env python3
from socketIO_client_nexus import SocketIO, LoggingNamespace

def on_connect():
    print('connect')

def on_disconnect():
    print('disconnect')

def on_reconnect():
    print('reconnect')

def on_Response_B(*args):
    print('I got a reponse from server for event_B', args)

def on_Response_C(*args):
    print('I got a reponse from server for event_C', args)


def FuncA():
    print('FuncA: Now I got a message from server.')

socketIO = SocketIO('localhost', 3000, LoggingNamespace)
socketIO.on('connect', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)
socketIO.on('event_A', FuncA)
# Listen
socketIO.on('Response_B', on_Response_B)
socketIO.emit('event_B')
socketIO.emit('event_B')
socketIO.wait(seconds=1)

# Stop listening
socketIO.off('Response_B')
socketIO.emit('event_B')
socketIO.wait(seconds=1)

# Listen only once
socketIO.once('Response_C', on_Response_C)
socketIO.emit('event_C')  # Activate aaa_response
socketIO.emit('event_C')  # Ignore
socketIO.wait(seconds=1)
