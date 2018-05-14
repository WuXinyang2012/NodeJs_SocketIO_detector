#! /usr/bin/env python3
import os,sys
import time

write_path = "/tmp/pipe.out"

wf = os.open(write_path,  os.O_CREAT | os.O_SYNC  | os.O_RDWR)
if len(sys.argv) == 1:
    msg = b"inference "
    len_send = os.write(wf, msg)
elif sys.argv[1] == '-e':
    msg = b"exit "
    len_send = os.write(wf, msg)


os.close(wf)