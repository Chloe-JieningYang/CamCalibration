import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer

# 计算 UTM 带号（UTM 带号计算公式）
def get_utm_epsg(lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone}"

def solve_pnp_from_files(gps_file, txt_file, camera_intrinsic, dist_coeffs, reference_point=None):
    """
    读取 3D 物体点 (CSV) 和 2D 图像点 (TXT)，使用 EPnP 求解位姿

    Args:
        csv_file (str): 3D 物体点的 CSV 文件路径 (格式: x,y,z)
        txt_file (str): 2D 图像点的 TXT 文件路径 (格式: x,y)
        camera_intrinsic (np.ndarray): 3×3 相机内参矩阵
        dist_coeffs (np.ndarray): 1×5 畸变系数

    Returns:
        tuple: (rvec, extrinsic, projection_matrix)
            rvec: 旋转向量 (3×1)
            extrinsic: 外参矩阵 (3×4)
            projection_matrix: 投影矩阵 (3×4)
    """
    # 读取 3D 物体点
    # obj_points = pd.read_csv(csv_file, header=None).values.astype(np.float32)
    column_names = ["lat", "lon"]
    gps_data = pd.read_csv(gps_file, header=None, usecols=[0, 1], names=column_names)
    gps_data = gps_data.to_numpy()

    # 创建 UTM 转换器（以第一行数据计算 UTM 带号）
    epsg_utm = get_utm_epsg(gps_data[0, 1])
    transformer = Transformer.from_crs("EPSG:4326", epsg_utm)

   # 初始化 N×3 矩阵
    obj_points = np.ones((gps_data.shape[0], 3)) # z轴赋值为1，为了方便scale

    # 转化成UTM
    obj_points[:, :2] = np.array(transformer.transform(gps_data[:, 0], gps_data[:, 1])).T

    # 确定参考点 (lat, lon, alt)
    if reference_point is None:
        reference_point = gps_data[0] # 默认使用第一行数据
    ref_lat, ref_lon = reference_point

    # 计算参考点的 UTM 坐标
    ref_x, ref_y = transformer.transform(ref_lat, ref_lon)
    ref_utm = np.array([ref_x, ref_y, 0])  # 参考 UTM 坐标

    # 归一化（减去参考点）
    obj_points -= ref_utm

    # 读取 2D 图像点
    img_points = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

    # 对图像点去畸变
    undistorted_points = cv2.undistortPoints(img_points.reshape(-1, 1, 2), camera_intrinsic, dist_coeffs)
    img_points = undistorted_points.reshape(-1, 2)

    # 计算 EPnP
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_intrinsic, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise ValueError("EPnP 求解失败")

    # 计算旋转矩阵
    rmat, _ = cv2.Rodrigues(rvec)

    # 构造外参矩阵 [R | T]
    extrinsic = np.hstack((rmat, tvec))

    # 计算投影矩阵 P = K * [R | T]
    projection_matrix = camera_intrinsic @ extrinsic

    return rvec, extrinsic, projection_matrix, ref_utm

# 示例使用
camera_intrinsic = np.array([[4324.24, 0, 959.5], [0, 4621.14, 539.5], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([-8.8e-6, 0, -0.00146, 0.00122, 0], dtype=np.float64)

rvec, extrinsic, projection_matrix, ref_utm= solve_pnp_from_files("label_data/gps.csv", "label_data/images.txt", camera_intrinsic, dist_coeffs)

print("Rotation Vector:\n", rvec)
print("\nExtrinsic Matrix:\n", extrinsic)
print("\nProjection Matrix:\n", projection_matrix)
print("\nReference Point:\n",ref_utm)
