# Author: Kylin
# Date: 2023-11-12 22:09:44
# File Name: offlinecali01
# Aim for: offline camera calibration
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 棋盘格的内部尺寸（单位：毫米）
square_size = 12.0

# 准备棋盘格角点的坐标
chessboard_size = (10, 7)  # 棋盘格内部的角点数量
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 存储检测到的角点和对应的物体点
objpoints = []  # 3D物体点
imgpoints = []  # 2D图像点

# 获取目录中的所有JPEG图像文件
image_dir = r'.\image'  # 设置为你的图像目录路径

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

# 打开每个图像文件并执行相机标定
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像文件: {image_file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", gray)  # check gray image
    # cv2.waitKey(500)  # 显示图像 500 毫秒
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 在图像上绘制角点
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

        # 显示带有角点的图像（可选）
        cv2.imshow('Chessboard', image)
        cv2.waitKey(30)  # 显示图像 500 毫秒

# 执行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("相机内部参数（相机矩阵）:")
    print(mtx)
    print("\n畸变系数:")
    print(dist)
else:
    print("标定失败")

# 关闭所有打开的窗口
cv2.destroyAllWindows()

# 检查标定后的重投影误差
# 计算每个图像上每个点的重投影误差
reprojection_errors = []
# 计算观测点和重投影点之间的坐标差异，并保留方向
for i in range(len(objpoints)):
    _, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
    imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx, dist)

    errors = []  # 保存每个目标点的重投影误差
    for j in range(len(imgpoints[i])):
        observed_point = imgpoints[i][j]
        reprojected_point = imgpoints_reprojected[j]
        dx = reprojected_point[0][0] - observed_point[0][0]
        dy = reprojected_point[0][1] - observed_point[0][1]
        errors.append((dx, dy))  # 保存坐标差异

    reprojection_errors.append(errors)  # 将每个图像上的重投影误差列表添加到总列表中

# 绘制重投影误差的直方图
# 将所有 dx 和 dy 提取出来
all_dx = [error[0] for errors in reprojection_errors for error in errors]
all_dy = [error[1] for errors in reprojection_errors for error in errors]

# 计算 dx 和 dy 的均方根误差
rmse_dx = np.sqrt(np.mean(np.square(all_dx)))
rmse_dy = np.sqrt(np.mean(np.square(all_dy)))

# 绘制 dx 和 dy 的直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(all_dx, bins=20, color='blue', alpha=0.7)
plt.xlabel('dx (Reprojection Error)')
plt.ylabel('Frequency')
plt.title('dx (Reprojection Error) Histogram')
plt.text(0.05, 0.9, f'RMSE: {rmse_dx:.2f} pixel', transform=plt.gca().transAxes, color='blue')

plt.subplot(1, 2, 2)
plt.hist(all_dy, bins=20, color='red', alpha=0.7)
plt.xlabel('dy (Reprojection Error)')
plt.ylabel('Frequency')
plt.title('dy (Reprojection Error) Histogram')
plt.text(0.05, 0.9, f'RMSE: {rmse_dy:.2f} pixel', transform=plt.gca().transAxes, color='red')

plt.show()
