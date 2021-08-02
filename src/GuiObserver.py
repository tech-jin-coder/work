# This sample code uses the Appium python client
# pip install Appium-Python-Client
# Then you can paste this into a file and simply run with Python

#%%
import argparse
from appium import webdriver
from time import sleep
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
class GuiObserver:
    
    def __init__(self , deviceName , appPackage , appActivity , AndroidVersion):
        self.caps = {}
        self.caps["platformName"] = "Android"
        self.caps["platformVersion"] = AndroidVersion
        self.caps["deviceName"] = deviceName
        self.caps["appPackage"] = appPackage
        # self.caps["appPackage"] = "org.liberty.android.fantastischmemo"
        self.caps["appActivity"] = appActivity
        self.caps["noReset"] = True
        # self.caps["appActivity"] = "org.liberty.android.fantastischmemo.ui.AnyMemo"
        self.caps['unicodeKeyboard'] = True
        self.caps['resetKeyboard'] = True
        self.caps["ensureWebviewsHavePages"] = True
        self.caps["newCommandTimeout"] = 2000

    def permission_choose_fun(self,driver):
        # 权限弹框处理
        while True:
            loc =  "//android.widget.Button[contains(@text,'允许')]"
            loc1 =  "//android.widget.Button[contains(@text,'确定')]"
            if '允许' in driver.page_source:
                driver.find_element_by_xpath(loc).click()
            elif '确定' in driver.page_source:
                
                driver.find_element_by_xpath(loc1).click()
            else:
                print("权限处理结束")
                break
            time.sleep(1)

    def get_driver(self):
        driver = webdriver.Remote("http://localhost:4723/wd/hub", self.caps)
        time.sleep(1)
        self.permission_choose_fun(driver)
        return driver

    def get_screenshot(self,driver):
        filename = 'img/1.png'
        driver.get_screenshot_as_file(filename)
        img = Image.open(filename)
        img = img.convert("RGB")
        return img  

    def getPage(self,driver):
        #获取当前页面的源。
        time.sleep(0.1)
        page = driver.page_source
        return page
        


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deviceName', default="192.168.194.105:5555" , help='device_number_str')
    args = parser.parse_known_args()[0]
    guiobserver = GuiObserver(args.deviceName)

# %%
