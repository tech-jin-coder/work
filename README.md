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
#calculate coverage
1. We use jacoco to calculate coverage.
2. You can download the APP source code in here.
  https://sstl-maprl.coding.net/s/e0de7635-5861-4e29-9ed3-49bebbc03064
  https://sstl-maprl.coding.net/s/3f9633fe-4ea9-49a1-b33f-76e940bdcbaf
  https://sstl-maprl.coding.net/s/df201a13-7ad0-4a18-8d32-76aa5174d5ba
  https://sstl-maprl.coding.net/s/677884ef-7031-4a0b-ab5d-f51b563fcd0a
  https://sstl-maprl.coding.net/s/91396f50-6d44-4a4e-a51f-ffe9f41effda
4. After testing ,run this code to get the coverage.ec
  ```bash
   adb pull mnt/sdcard/coverage.ec C:\Users\user\Desktop\testReport\jacoco
  ```
4. Put the EC file under “build/outputs/code coverage/connected” in the source code.
5. Use Android Studio or Gradle to Run JacocoTestReport,then you can find the result in the "build/reports/jacoco/jacocoTestReport".
