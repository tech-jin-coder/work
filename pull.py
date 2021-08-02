import datetime
import os
import time
t = datetime.datetime.now()
pos = 0
while 1:
    now = datetime.datetime.now()
    if ((now - t).seconds % 120)==0 and (now - t).seconds > 0 and pos <15: 
        cmd = "adb pull /sdcard/coverage.ec E:/自动化测试/mywork_not_use_ae/co%d"%pos
        os.system(cmd)
        time.sleep(1)
        pos+=1
    elif ((now - t).seconds % 300)==0 and (now - t).seconds > 0 and pos >= 15:
        cmd = "adb pull /sdcard/coverage.ec E:/自动化测试/mywork_not_use_ae/co%d"%pos
        os.system(cmd)
        time.sleep(1)
        pos+=1
    if (now - t).seconds > 3600:
        os.pause()
