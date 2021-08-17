# Install Some Software
1. install [Appium](http://appium.io/)
2. install [Geymotion](https://www.genymotion.com/) or ohter Android emulator.
3. open Geymotion and create a virtual device.'
# Installation
1. Install Requirements
  ```shell
   pip install -r requirements.txt
  ```
2. Open appium software and start the service.
3. Install the app under APK folder into the phone。
  ```shell
   adb install ./APK/OmniNotes-playDebug-6.1.0Alpha1.apk
  ```
4. get your device address  
  ```shell
   adb devices
  ``` 
5. Start Runing
  ```shell
   python main.py 192.168.242.101:5555 cuda feio 8.0.0
  ```
6. You can change "192.168.242.101:5555" to your device address. 
7. "cuda" means to use "GPU".It can be repaced by "cpu".
8. "feio" is the app name,it can be repaced by the following word：AnyMemo；xinwen；kaiyan；xiaoqiu
9. "8.0.0" is the Android version number of the tested mobile phone, please modify it yourself.
