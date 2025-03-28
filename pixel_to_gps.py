import numpy as np
import pandas as pd
import os
from pyproj import Transformer
import cv2


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
utm_matrix = np.zeros((gps_data.shape[0], 3))

# 转化成UTM
utm_matrix[:, :2] = np.array(transformer.transform(gps_data[:, 0], gps_data[:, 1])).T

# print(utm_matrix)


# 读取像素数据，x,y
# label_folder = "labels"
# for i in range(10):
#     file_path = os.path.join(label_folder, f"{i}.txt")
#     # 确保文件存在
#     if os.path.exists(file_path):
#         # 读取文件内容
#         with open(file_path, 'r') as file:
#             content = file.read().strip()
#             # 假设文件内容是以逗号分隔的数值，转换为浮点数列表
#             data_row = np.array([float(num) for num in content.split()[1:]], dtype=float)
#             # 将这个列表添加到数据列表中
#             pixel_data_list.append(data_row)
#     else:
#         print(f"File {file_path} does not exist")

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

# 对像素坐标点去畸变
undistorted_points = cv2.undistortPoints(pixel_data.reshape(-1, 1, 2), camera_intrinsic, dist_coeffs)
pixel_data = undistorted_points.reshape(-1, 2)

ones_column = np.ones((utm_matrix.shape[0], 1))
Y = np.concatenate((utm_matrix, ones_column), axis=1)

ones_column = np.ones((pixel_data.shape[0], 1))
X = np.concatenate((pixel_data, ones_column), axis=1)

A_estimated = np.linalg.lstsq(X, Y, rcond=None)[0]

print(
    "qA=Q, q represents image coordinate [u,v,1]; Q represents UTM coordinate [x,y,z,1]"
)
print(A_estimated)
predict_Q = X @ A_estimated
print("predicted Q:", predict_Q)
print("real Q:", Y)

mae_1 = np.mean(np.abs(predict_Q[:, 0:1] - Y[:, 0:1]))
print("mae:", mae_1)

###############################
## test & check
###############################
# image_test_file = "images_2.txt"
# with open(image_test_file, "r") as file:
#     pixel_test_data_list = [
#         [float(value) for value in line.strip().split(",")] for line in file
#     ]
# pixel_test_data_list = np.array(pixel_test_data_list)
 
# gps_test_file = "gps_2.CSV"

# column_names = ["lat", "lon"]
# gps_test_data = pd.read_csv(
#     gps_test_file, header=None, usecols=[0, 1], names=column_names
# )
# gps_test_data = gps_test_data.to_numpy()

# # 初始化 N×3 矩阵
# utm_test_matrix = np.zeros((gps_test_data.shape[0], 3))

# utm_test_matrix[:, :2] = np.array(
#     transformer.transform(gps_test_data[:, 0], gps_test_data[:, 1])
# ).T

# ones_column = np.ones((pixel_test_data_list.shape[0], 1))
# X_test = np.concatenate((pixel_test_data_list, ones_column), axis=1)

# ones_column = np.ones((utm_test_matrix.shape[0], 1))
# Y_test = np.concatenate((utm_test_matrix, ones_column), axis=1)

# Y_pred = X_test @ A_estimated
# mae = np.mean(np.abs(Y_test[:, 0:1] - Y_pred[:, 0:1]))
# print("mae:", mae)
