#%%
#import util
from Agent import *
from src.GuiObserver import GuiObserver
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import xml.etree.ElementTree as ET
import torch
import numpy as np
import sys
import datetime
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('deviceName', default="192.168.194.105:5555" , help='device_number_str')
    parser.add_argument('device', default="cuda" , help='cpu or cuda(gpu)')
    parser.add_argument('app', default="who")
    parser.add_argument('AndroidVersion',default="8.0.0")
    parser.add_argument('start_time',default="1")
    App = {
        "xinwen" : ("com.tiger.quicknews" , "com.tiger.quicknews.activity.WelcomeActivity_","新闻快讯"),
        "AnyMemo" : ("org.liberty.android.fantastischmemo" , "org.liberty.android.fantastischmemo.ui.AnyMemo","AnyMemo"),
        "who" : ("de.freewarepoint.whohasmystuff" , "de.freewarepoint.whohasmystuff.MainActivity","Who Has My Stuff?"),
        "kaiyan" : ("openeyes.dr.openeyes" , ".MainActivity","开眼短视频"),
        "feio" : ("it.feio.android.omninotes" , "it.feio.android.omninotes.MainActivity","Omni Notes"),
        "xiaoqiu":("com.ocnyang.qbox.app" , ".module.mains.MainsActivity","小秋魔盒"),
    }
    args = parser.parse_args()
    device = torch.device(args.device)
    AndroidVersion = args.AndroidVersion
    appPackage = App[args.app][0]
    appActivity = App[args.app][1]
    appName = App[args.app][2]
    start_date_str = args.start_time
    if(start_date_str.equals("1")):
        start_t = datetime.datetime.now()
        start_data_str = start_t.strftime(("%Y-%m-%d_%H:%M:%S"))
    start_date = datetime.datetime.strptime(start_date_str,("%Y-%m-%d_%H:%M:%S"))
    now = datetime.datetime.now()
    if (now - start_date).seconds > 1800:
        sys.exit(0)
#%%
    guiobserver = GuiObserver(args.deviceName , appPackage , appActivity , AndroidVersion)
#%%
    
#%%
    custom_transform = transforms.Compose([
                                       transforms.Resize((256,256)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])
    agent = Agent(custom_transform,appPackage,appActivity,appName,device)
#%%
    try:
        agent.train(150,400,guiobserver,device,start_date)
    except Exception as e:
        print(e)
#%%
 