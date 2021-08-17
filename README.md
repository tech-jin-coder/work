# work
1. >install [Appium](http://appium.io/)
2. >install [Geymotion](https://www.genymotion.com/) or ohter Android emulator.
3. >open Geymotion and create a virtual device.
4. pip install -r requirements.txt
5. 打开Appium软件，启动服务。
6. 将APK文件夹下的安装包安装到手机中。
7. python main.py 192.168.242.101:5555 cuda feio 8.0.0
8. 其中192.168.242.101:5555为adb devices中获取的device地址
9. cuda为使用GPU，可以改为cpu
10. feio为软件的名字,可以改为以下软件：AnyMemo；xinwen；kaiyan；xiaoqiu
11. 8.0.0位被测手机的安卓版本号，请自行修改。
