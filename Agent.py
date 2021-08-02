import numpy as np                             
from DDQN import DDQN
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from util import *
import random
import sys
import string
from appium.webdriver.common.touch_action import TouchAction
from appium.webdriver.connectiontype import ConnectionType
from random import randint
import math
import torch
import os
import copy
import datetime
import time
def ranstr(num):
    return "test"

def cos(array1,array2):
    array1 = 255 - array1.to("cpu").view(1,-1).float()
    array2 = 255 - array2.to("cpu").view(1,-1).float()
    #print(array1)
    #print(array2)
    ssim = torch.cosine_similarity(array1,array2)
    return ssim

class Agent(object):
    def __init__(self,custom_transform,package,appActivity,appName,device,action_space = 7):
        self.custom_transform = custom_transform
        self.package = package
        self.appActivity = appActivity
        self.appName = appName
        self.device = device
        
        self.action_space = action_space
        
        self.ddqn = DDQN(self.action_space,self.device,self.appName)
        self.img_alpha = 0.8
        self.memory1 = {}
    
    

    def train(self,epsilon,step,guiobserver,device,start_date):
        pos = 0
        driver = guiobserver.get_driver()
        page = guiobserver.getPage(driver)
        width = int(driver.get_window_size()['width'])
        height = int(driver.get_window_size()['height'])
        #(1,1,256,256)
        util = Util(self.custom_transform,self.device,width,height)
        state_img = util.get_state_img(page)
        state_hx = torch.zeros((1,512)).to(device)
        state_cx = torch.zeros((1,512)).to(device)
        state = (state_img, (state_hx, state_cx))
        ac = driver.current_activity
        init_state = (state_img, (state_hx, state_cx))

        flag_net = 0
        flag_rot = 0

        for i in range(epsilon):
            while('com.android' not in driver.page_source and 'com.huawei.android' not in driver.page_source and i!=0):
                driver.press_keycode(3)
            if i!=0:
                driver.start_activity(self.package,self.appActivity)
            for j in range(step):
                now = datetime.datetime.now()
                if (now - start_date).seconds > 1800:
                    return
                ac = driver.current_activity
                state1 = copy.deepcopy(state)
                if(j !=0 and j%50 == 0):
                    #print(os.path)
                    tmps = self.appName
                    torch.save(self.ddqn.eval_net.state_dict(),'%snet.pth'%tmps[0])
                list1 = driver.find_elements_by_class_name("android.widget.EditText")
                for x in list1:
                    if x.get_attribute('text') == '' or x.get_attribute('text')[0] != 'a':
                        x.click()
                        context2=x.get_attribute('text')
                        driver.keyevent(123)
                        for i in range(0,len(context2)):
                            driver.keyevent(67)
                        x.send_keys("a"+ranstr(5))
                #perform(click(),closeSoftKeyboard())
                # tmp_net = np.random.uniform()
                # if tmp_net > 0.9:
                #     driver.set_network_connection(ConnectionType.AIRPLANE_MODE)
                #     flag_net = 1
                # tmp_rot = np.random.uniform()
                # if tmp_rot > 0.9:
                #     driver.orientation = "LANDSCAPE"
                #     time.sleep(0.5)
                #     if driver.orientation == "LANDSCAPE":
                #         flag_rot = 1
                action,location,(hx,cx) = self.ddqn.choose_action(state)
                location_x = int(float(width)/256 * location[0])
                location_y = int(float(height)/256 * location[1])
                print("action")
                print(action)
                print("location")
                print((location_x,location_y))
                
                if flag_rot == 1:
                    location_t = location_x
                    location_x = location_y
                    location_y = location_t

                try:
                    if action == 0:
                        TouchAction(driver).tap(x=location_x,y=location_y).perform()
                    
                    if action == 1:
                        TouchAction(driver).long_press(x=location_x,y=location_y).release().perform()
                    
                    if action == 2:
                        TouchAction(driver).tap(x=location_x,y=location_y).perform()
                        loc =  "//android.widget.Button[contains(@text,'确定')]"
                        loc1 =  "//android.widget.Button[contains(@text,'submit')]"
                        loc2 =  "//android.widget.Button[contains(@text,'ADD')]"
                        if '确定' in driver.page_source:
                            driver.find_element_by_xpath(loc).click()
                        elif  'submit' in driver.page_source:
                            driver.find_element_by_xpath(loc1).click()
                        elif 'ADD' in driver.page_source:
                            driver.find_element_by_xpath(loc2).click()

                    if action == 3 or action == 4 or action == 5 or action == 6:
                        switch = {
                            3:lambda x,y:((x,100 if y>=200 else x,y)),
                            4:lambda x,y:((x,height - 100 if y<=height - 200 else x,y)),
                            5:lambda x,y:((100,y if x>=200 else x,y)),
                            6:lambda x,y:((width - 100,y if x<=width - 200  else x,y))
                        }
                        if flag_rot == 0:
                            scroll_loc = switch[action](location_x,location_y)
                            driver.swipe(location_x,location_y,scroll_loc[0],scroll_loc[1])
                        else:
                            scroll_loc = switch[action](location_y,location_x)
                            driver.swipe(location_x,location_y,scroll_loc[1],scroll_loc[0])

                    if action == 7:
                        driver.press_keycode(4)                
                    if action == 8:
                        driver.press_keycode(4)
                        driver.press_keycode(4)
                        driver.press_keycode(4)

                except Exception as e:
                    print(e)

                # if flag_net == 1:
                #     driver.set_network_connection(ConnectionType.ALL_NETWORK_ON)
                #     flag_net = 0
                    

                # if flag_rot == 1:
                #     location_t = location_x
                #     location_x = location_y
                #     location_y = location_t
                #     driver.orientation = "PORTRAIT"
                #     time.sleep(0.5)
                #     flag_rot = 0
                new_page = guiobserver.getPage(driver)
                new_state_encode_img = util.get_state_img(new_page)
                new_state = (new_state_encode_img,(hx.data,cx.data))
                similar = cos(state1[0][0] , new_state_encode_img[0])
                #print(similar)
                ac = driver.current_activity
                r_img = 0.0

                if  similar < self.img_alpha:
                    r_img = -1
                    if similar < 0.4:
                        r_img = 0
                else:
                    r_img = -20.0

                if ac in self.memory1 :
                    self.memory1[ac] += 1
                else:
                    self.memory1[ac] = 1             

                r_cnt = 1.0/(int)(self.memory1[ac]) if self.memory1[ac] < 10 else -2
                if int(self.memory1[ac] ) == 1:
                    r_cnt = 10

                reward = r_cnt + r_img
                if 'com.android' in driver.page_source or 'com.huawei.android' in driver.page_source:
                    reward = -30.0
                print("reward")
                print("r_cnt = %f,r_img = %f,reward = %f" %(r_cnt,r_img,reward))
                if(action != 8 and action != 7):
                    self.ddqn.store_transition(state1,(action,location[0]*256+location[1]),reward,new_state)
                state = copy.deepcopy(new_state)
                page = new_page
                if self.ddqn.memory_counter > 32:
                    self.ddqn.learn()
                if 'com.android' in driver.page_source or 'com.huawei.android' in driver.page_source:
                    #print(1)
                    driver.press_keycode(3)
                    #print(2)
                    while(self.appName not in driver.page_source):
                        #print(3)
                        driver.swipe(793,679,510,671,100)
                        page = guiobserver.getPage(driver)
                    el1 = driver.find_element_by_accessibility_id(self.appName)
                    el1.click()
                    continue
                elif self.package not in driver.page_source:
                    while(self.package not in driver.page_source and   'com.android' not in driver.page_source and 'com.huawei.android' not in driver.page_source):
                        driver.press_keycode(3)
                    if 'com.android' in driver.page_source or 'com.huawei.android' in driver.page_source:
                        driver.press_keycode(3)
                        while(self.appName not in driver.page_source):
                            driver.swipe(793,679,510,671,100)
                            page = guiobserver.getPage(driver)
                        el1 = driver.find_element_by_accessibility_id(self.appName)
                        el1.click()
                    continue
            
                

                
                

                
