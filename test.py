#%%
from util import *
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
    parser.add_argument('--deviceName', default="192.168.242.101:5555" , help='device_number_str')
    parser.add_argument('--device', default="cuda" , help='cpu or cuda(gpu)')
    parser.add_argument('--app', default="feio")
    #parser.add_argument('start_time')
    App = {
        "xinwen" : ("com.tiger.quicknews" , "com.tiger.quicknews.activity.WelcomeActivity_","新闻快讯"),
        "AnyMemo" : ("org.liberty.android.fantastischmemo" , "org.liberty.android.fantastischmemo.ui.AnyMemo","AnyMemo"),
        "who" : ("de.freewarepoint.whohasmystuff" , "de.freewarepoint.whohasmystuff.MainActivity","Who Has My Stuff?"),
        "kaiyan" : ("openeyes.dr.openeyes" , ".MainActivity","开眼短视频"),
        "feio" : ("it.feio.android.omninotes" , "it.feio.android.omninotes.MainActivity","Omni Notes"),
    }
    args =parser.parse_known_args()[0]
    device = torch.device(args.device)
    appPackage = App[args.app][0]
    appActivity = App[args.app][1]
    appName = App[args.app][2]
    # start_date_str = args.start_time
    # start_date = datetime.datetime.strptime(start_date_str,("%Y-%m-%d_%H:%M:%S"))
    # now = datetime.datetime.now()
    # if (now - start_date).seconds > 1800:
    #     sys.exit(0)

#%%
    guiobserver = GuiObserver(args.deviceName , appPackage , appActivity)
#%%
    
#%%
    custom_transform = transforms.Compose([
                                       transforms.Resize((256,256)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])
    agent = Agent(custom_transform,appPackage,appActivity,appName,device)

#%%
    driver = guiobserver.get_driver()
#%%
    page = guiobserver.getPage(driver)
    print(page)
#%%
    import imp
    import util
    imp.reload(util)
    from util import *
    width = int(driver.get_window_size()['width'])
    height = int(driver.get_window_size()['height'])
    util1 = Util(custom_transform,device,width,height)
    print(width)
    print(height)
#%%
    print(util1.get_state_img(page))
# %%
    guiobserver.get_screenshot(driver)
# %%
    TouchAction(driver).tap(x=55,y=120).perform()
# %%
    agent.train(150,400,guiobserver,device,datetime.datetime.now())
# %%
