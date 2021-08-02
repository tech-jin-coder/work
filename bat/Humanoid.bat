start cmd /k "call activate tensorflowone && cd .. && cd Humanoid_master && python agent.py -c config.json"
ping -n 10 127.1 >nul 2>nul
call activate tensorflowone
droidbot -a C:\Users\13703\Desktop\asd\OmniNotes-playDebug-6.1.0.apk -o E:\自动化测试\output -humanoid localhost:50405
