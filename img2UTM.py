import numpy as np
import cv2

def image_to_utm(image_points: np.ndarray, camera_intrinsic: np.ndarray, dist_coeffs: np.ndarray,
                 rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """
    将像素坐标转换为 UTM 坐标。

    参数：
        image_points (np.ndarray): 形状 (N, 2) 的像素坐标数组
        camera_intrinsic (np.ndarray): 相机内参矩阵 (3x3)
        dist_coeffs (np.ndarray): 畸变系数 (1x5 或 None)
        rotation_vector (np.ndarray): 旋转向量 (3x1)
        translation_vector (np.ndarray): 平移向量 (3x1)

    返回：
        np.ndarray: 形状 (N, 3) 的 UTM 坐标数组
    """
    # 计算旋转矩阵
    R, _ = cv2.Rodrigues(rotation_vector)
    
    # 计算 [R|T] 外参矩阵
    Rt = np.hstack((R, translation_vector))  # 3x4 变换矩阵
    
    # 计算相机内参的逆矩阵
    K_inv = np.linalg.inv(camera_intrinsic)

    # 将像素坐标转换为齐次坐标 (u, v, 1)
    ones = np.ones((image_points.shape[0], 1))
    image_points_homogeneous = np.hstack((image_points, ones)).T  # 3xN

    # 去畸变
    if dist_coeffs is not None and len(dist_coeffs) > 0:
        image_points = cv2.undistortPoints(image_points.reshape(-1, 1, 2), camera_intrinsic, dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        image_points_homogeneous[:2, :] = image_points.T  # 更新去畸变的点

    # 计算世界坐标（反投影）
    world_coords = []
    for i in range(image_points_homogeneous.shape[1]):
        uv1 = image_points_homogeneous[:, i]  # 3x1
        cam_coords = K_inv @ uv1  # 归一化相机坐标系中的点
        world_coord = np.linalg.inv(R) @ (cam_coords - translation_vector.flatten())  # 计算世界坐标
        world_coords.append(world_coord)

    return np.array(world_coords)  # (N, 3)

# 示例数据
image_points = np.array([[500, 300], [600, 400]])  # 假设是像素坐标
camera_intrinsic = np.array([[1000, 0, 512], [0, 1000, 384], [0, 0, 1]])  # 内参矩阵
dist_coeffs = np.zeros(5)  # 假设无畸变
rotation_vector = np.array([[0.1], [0.2], [0.3]])  # PnP 计算出的旋转向量
translation_vector = np.array([[10], [20], [30]])  # PnP 计算出的平移向量

# 转换
utm_coords = image_to_utm(image_points, camera_intrinsic, dist_coeffs, rotation_vector, translation_vector)
print(utm_coords)  # 形状为 (N, 3)