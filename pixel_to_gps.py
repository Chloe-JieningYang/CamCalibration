import numpy as np
import pandas as pd
import os
from pyproj import Transformer
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


gps_file = "label_data/gps.CSV"

# 读取gps数据，包括纬度，经度，海拔
column_names = ["lat", "lon"]
gps_data = pd.read_csv(gps_file, header=None, usecols=[0, 1], names=column_names)
gps_data = gps_data.to_numpy()


# 计算 UTM 带号（UTM 带号计算公式）
def get_utm_epsg(lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone}"


# 创建 UTM 转换器（以第一行数据计算 UTM 带号）
epsg_utm = get_utm_epsg(gps_data[0, 1])
transformer = Transformer.from_crs("EPSG:4326", epsg_utm)

# 初始化 N×3 矩阵
utm_matrix = np.ones((gps_data.shape[0], 3))

# 转化成UTM
utm_matrix[:, :2] = np.array(transformer.transform(gps_data[:, 0], gps_data[:, 1])).T

# print(utm_matrix)

image_file = "label_data/images.txt"
with open(image_file, "r") as file:
    pixel_data_list = [
        [float(value) for value in line.strip().split(",")] for line in file
    ]

pixel_data = np.array(pixel_data_list)[:, [0, 1]]

camera_intrinsic=np.array([[2.51112220e+03, 0.00000000e+00, 9.71406297e+02],
 [0.00000000e+00, 2.51206946e+03, 5.73149627e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.array([[-8.37716727e-01, 1.84486584e+00, 3.00158084e-03, 5.51622873e-03, -3.21176567e+00]], dtype=np.float32)

# 去畸变
undistorted_points = cv2.undistortPoints(pixel_data.reshape(-1, 1, 2), camera_intrinsic, dist_coeffs)
undistorted_pixel = cv2.convertPointsToHomogeneous(undistorted_points).reshape(-1, 3)
pixel_data = (camera_intrinsic @ undistorted_pixel.T).T[:, :2]

# print('\npixel_data: ',pixel_data)

H, mask = cv2.findHomography(pixel_data, utm_matrix[:, :2], method=cv2.RANSAC)

# 用H进行像素到UTM的预测
pixel_data_hom = np.hstack((pixel_data, np.ones((pixel_data.shape[0], 1))))
predict_Q = (pixel_data_hom @ H.T)
predict_Q = predict_Q[:, :2] / predict_Q[:, 2, np.newaxis]

# 计算误差
mae = np.mean(np.linalg.norm(predict_Q - utm_matrix[:, :2], axis=1))
print(f"Homography-based MAE: {mae:.3f} meters")

# 画拟合点 vs 真实点的对比
plt.figure(figsize=(12, 8))

# 绘制真实 UTM 坐标（蓝色圆点）
plt.scatter(utm_matrix[:, 0], utm_matrix[:, 1], c='blue', label='Ground Truth (UTM)', s=50)

# 绘制预测的 UTM 坐标（红色叉）
plt.scatter(predict_Q[:, 0], predict_Q[:, 1], c='red', marker='x', label='Predicted (Homography)', s=50)

# 为每个点画一条线：真实点 → 预测点
for i in range(len(utm_matrix)):
    plt.plot([utm_matrix[i, 0], predict_Q[i, 0]],
             [utm_matrix[i, 1], predict_Q[i, 1]],
             color='gray', linestyle='--', linewidth=0.8)

# 图例、标题、坐标轴
plt.legend()
plt.title(f'Pixel ➔ UTM Mapping via Homography\nMAE: {mae:.3f} meters', fontsize=16)
plt.xlabel('UTM Easting', fontsize=14)
plt.ylabel('UTM Northing', fontsize=14)
plt.axis('equal')   # 保证x和y比例一样
plt.grid(True)
plt.show()

# # UTM坐标归一化
# utm_min = np.min(utm_matrix[:, :2], axis=0)
# utm_max = np.max(utm_matrix[:, :2], axis=0)
# utm_center = (utm_max + utm_min) / 2
# utm_scale = (utm_max - utm_min) / 2

# # 像素坐标归一化
# pixel_min = np.min(pixel_data, axis=0)
# pixel_max = np.max(pixel_data, axis=0)
# pixel_center = (pixel_max + pixel_min) / 2
# pixel_scale = (pixel_max - pixel_min) / 2

# # 归一化UTM坐标
# utm_normalized = np.ones_like(utm_matrix)
# utm_normalized[:, :2] = (utm_matrix[:, :2] - utm_center) / utm_scale

# # 归一化像素坐标
# pixel_normalized = (pixel_data - pixel_center) / pixel_scale

# # 构建归一化后的矩阵
# ones_column = np.ones((utm_normalized.shape[0], 1))
# Y = np.concatenate((utm_normalized, ones_column), axis=1)

# ones_column = np.ones((pixel_normalized.shape[0], 1))
# X = np.concatenate((pixel_normalized, ones_column), axis=1)

# # 使用最小二乘法求解初始转换矩阵
# A_estimated = np.linalg.lstsq(X, Y, rcond=None)[0]

# def optimize_transform_lm(X, Y, A_initial):
#     """使用LM算法优化转换矩阵，添加更多控制参数"""
#     def error_function(params):
#         A = params.reshape(3, 4)
#         predicted = X @ A
        
#         # 位置误差
#         position_errors = (predicted[:, :2] - Y[:, :2]).ravel()
        
#         # 添加小的正则化项，防止参数过大
#         regularization = 1e-6 * np.linalg.norm(A)
        
#         return np.concatenate([
#             position_errors,
#             [regularization]
#         ])
    
#     result = least_squares(
#         error_function, 
#         A_initial.ravel(), 
#         method='lm',          # Levenberg-Marquardt算法
#         max_nfev=10000,       # 增加最大迭代次数
#         ftol=1e-12,          # 函数收敛容差更严格
#         xtol=1e-12,          # 参数收敛容差更严格
#         gtol=1e-12,          # 梯度收敛容差更严格
#         x_scale='jac',        # 自动缩放参数
#         loss='linear',        # 使用线性损失函数
#         verbose=2            # 显示优化过程
#     )
    
#     if not result.success:
#         print("Warning: Optimization did not converge!")
#         print("Status:", result.status)
#         print("Message:", result.message)
    
#     return result.x.reshape(3, 4), result.cost

# # 使用优化后的函数
# A_optimized, final_cost = optimize_transform_lm(X, Y, A_estimated)

# print(f"Final optimization cost: {final_cost}")


# # 将预测结果转换回原始尺度
# def transform_points(points, A, pixel_center, pixel_scale, utm_center, utm_scale):
#     # 归一化输入点
#     points_normalized = (points - pixel_center) / pixel_scale
    
#     # 添加齐次坐标
#     ones = np.ones((points.shape[0], 1))
#     points_homogeneous = np.concatenate((points_normalized, ones), axis=1)
    
#     # 使用转换矩阵
#     predicted_normalized = points_homogeneous @ A
    
#     # 转换回原始尺度
#     predicted = np.zeros_like(predicted_normalized)
#     predicted[:, :2] = predicted_normalized[:, :2] * utm_scale + utm_center
#     predicted[:, 2:] = predicted_normalized[:, 2:]
    
#     return predicted

# # 计算并比较结果
# predict_Q_original = transform_points(pixel_data, A_estimated, pixel_center, pixel_scale, utm_center, utm_scale)
# predict_Q_optimized = transform_points(pixel_data, A_optimized, pixel_center, pixel_scale, utm_center, utm_scale)

# # 计算误差
# mae_original = np.mean(np.abs(predict_Q_original[:, :2] - utm_matrix[:, :2]))
# mae_optimized = np.mean(np.abs(predict_Q_optimized[:, :2] - utm_matrix[:, :2]))

# print(f"Original MAE in meters: {mae_original}")
# print(f"Optimized MAE in meters: {mae_optimized}")

# # 可视化结果
# plt.figure(figsize=(12, 6))

# plt.subplot(121)
# plt.scatter(utm_matrix[:, 0], utm_matrix[:, 1], c='blue', label='Ground Truth')
# plt.scatter(predict_Q_original[:, 0], predict_Q_original[:, 1], c='red', label='Original')
# plt.legend()
# plt.title(f'Original (MAE: {mae_original:.2f}m)')

# plt.subplot(122)
# plt.scatter(utm_matrix[:, 0], utm_matrix[:, 1], c='blue', label='Ground Truth')
# plt.scatter(predict_Q_optimized[:, 0], predict_Q_optimized[:, 1], c='green', label='Optimized')
# plt.legend()
# plt.title(f'Optimized (MAE: {mae_optimized:.2f}m)')

# plt.tight_layout()
# plt.show()

# # 6. 保存优化后的结果
# predict_Q = predict_Q_optimized
# print("\nFinal predicted UTM coordinates:")
# print(predict_Q)

###############################
## test & check
###############################
# ===================== 读取测试数据 ======================
# 1. 读取测试像素点
image_test_file = "label_data/images_2.txt"
with open(image_test_file, "r") as file:
    pixel_test_data_list = [
        [float(value) for value in line.strip().split(",")] for line in file
    ]
pixel_test_data = np.array(pixel_test_data_list)[:, [0, 1]]  # 只取xy

# 2. 读取测试GPS点
gps_test_file = "label_data/gps_2.CSV"
column_names = ["lat", "lon"]
gps_test_data = pd.read_csv(
    gps_test_file, header=None, usecols=[0, 1], names=column_names
)
gps_test_data = gps_test_data.to_numpy()

# 3. 转换成UTM坐标
utm_test_matrix = np.zeros((gps_test_data.shape[0], 3))
utm_test_matrix[:, :2] = np.array(
    transformer.transform(gps_test_data[:, 0], gps_test_data[:, 1])
).T

# ===================== 测试像素预处理（去畸变） ======================
# 测试集像素点也要做去畸变，和训练保持一致！
undistorted_points_test = cv2.undistortPoints(
    pixel_test_data.reshape(-1, 1, 2), 
    camera_intrinsic, 
    dist_coeffs
)
undistorted_pixel_test = cv2.convertPointsToHomogeneous(undistorted_points_test).reshape(-1, 3)
pixel_test_data = (camera_intrinsic @ undistorted_pixel_test.T).T[:, :2]

# ===================== 用Homography推理预测 ======================
# 构建齐次坐标
pixel_test_hom = np.hstack((pixel_test_data, np.ones((pixel_test_data.shape[0], 1))))  # (N,3)

# 用训练阶段求得的Homography矩阵 H 进行推理
predict_Q_test = (pixel_test_hom @ H.T)  # (N,3)
predict_Q_test = predict_Q_test[:, :2] / predict_Q_test[:, 2, np.newaxis]  # 除以第三列归一化

# ===================== 测试误差计算 ======================
errors_test = np.linalg.norm(predict_Q_test - utm_test_matrix[:, :2], axis=1)
mae_test = np.mean(errors_test)
max_error = np.max(errors_test)
median_error = np.median(errors_test)
std_error = np.std(errors_test)

print(f"Test Set MAE: {mae_test:.3f} meters")
print(f"Test Set Max Error: {max_error:.3f} meters")
print(f"Test Set Median Error: {median_error:.3f} meters")
print(f"Test Set Std Deviation: {std_error:.3f} meters")

# ===================== 画测试集拟合结果 ======================
plt.figure(figsize=(12, 8))
plt.scatter(utm_test_matrix[:, 0], utm_test_matrix[:, 1], c='blue', label='Ground Truth (Test UTM)', s=50)
plt.scatter(predict_Q_test[:, 0], predict_Q_test[:, 1], c='green', marker='x', label='Predicted (Homography)', s=50)

# 每个点连线
for i in range(len(utm_test_matrix)):
    plt.plot([utm_test_matrix[i, 0], predict_Q_test[i, 0]],
             [utm_test_matrix[i, 1], predict_Q_test[i, 1]],
             color='gray', linestyle='--', linewidth=0.8)

plt.legend()
plt.title(f'Homography Prediction on Test Set\nMAE: {mae_test:.3f} meters', fontsize=16)
plt.xlabel('UTM Easting')
plt.ylabel('UTM Northing')
plt.axis('equal')
plt.grid(True)
plt.show()
