#coding=UTF-8
import os
from time import time
import sys
import timeit
import datetime
if __name__ == '__main__':
    start_t = datetime.datetime.now()
    start_t_string = start_t.strftime(("%Y-%m-%d_%H:%M:%S"))
    while 1:
        now = datetime.datetime.now()
        if (now - start_t).seconds > 1800:
            break
        cmd = "python main.py 192.168.242.101:5555 cuda feio " + start_t_string
        #cmd = "python main.py NAB0220630017320 cuda AnyMemo " + start_t_string   #必须转义'\'
        #cmd = "python main.py NAB0220630017320 cuda who " + start_t_string
        #cmd = "python main.py NAB0220630017320 cuda feio " + start_t_string
        #cmd = "python main.py NAB0220630017320 cuda xinwen " + start_t_string
        #cmd = "python main.py NAB0220630017320 cuda kaiyan " + start_t_string
        #cmd = "python main.py NAB0220630017320 cuda xiaoqiu " + start_t_string
        #cmd = "python main.py 192.168.242.101:5555 cuda xiaoqiu " + start_t_string
        os.system(cmd)
