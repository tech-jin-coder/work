# work
1. 安装Appium.http://appium.io/
2. pip install -r requirements.txt
3. 打开Appium软件，启动服务。
4. 将APK文件夹下的安装包安装到手机中。
5. python main.py 192.168.242.101:5555 cuda feio 8.0.0
6. 其中192.168.242.101:5555为adb devices中获取的device地址
7. cuda为使用GPU，可以改为cpu
8. feio为软件的名字,可以改为以下软件：AnyMemo；xinwen；kaiyan；xiaoqiu
9. 8.0.0位被测手机的安卓版本号，请自行修改。
