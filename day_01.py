# Author: Kylin
# Date: 2023-11-12 20:32:31
# File Name: day_01
# Aim for:
import cv2
# 打开摄像头（默认是第一个摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()
image_counter = 0  # 用于计数的变量
while True:
    # 从摄像头捕获一帧图像
    ret, frame = cap.read()

    if not ret:
        print("无法捕获图像")
        break

    # 在窗口中显示捕获的图像
    cv2.imshow("Camera Feed", frame)

    # 如果按下键盘上的 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 如果按下 's' 键保存当前图像
    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_name = f"{image_counter:02d}.jpg"  # 格式化文件名，例如 01.jpg, 02.jpg, ...
        cv2.imwrite(image_name, frame)
        print(f"图像 {image_name} 已保存")
        image_counter += 1  # 增加计数器

# 释放摄像头资源
cap.release()

# 关闭所有打开的窗口
cv2.destroyAllWindows()

