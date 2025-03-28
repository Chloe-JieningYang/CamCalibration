import numpy as np
import cv2

def image_to_utm(image_points: np.ndarray,camera_intrinsic: np.ndarray, projection_matrix: np.ndarray, dist_coeffs: np.ndarray, depth=1) -> np.ndarray:
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
    
    # 去畸变
    if dist_coeffs is not None and len(dist_coeffs) > 0:
        image_points = cv2.undistortPoints(image_points.reshape(-1, 1, 2), camera_intrinsic, dist_coeffs)
        image_points = image_points.reshape(-1, 2)

    # 将去畸变后的像素坐标转换为齐次坐标 (u, v, 1)
    ones = np.ones((image_points.shape[0], 1))
    image_points_homogeneous = np.hstack((image_points, ones)).T  # 3xN

    # 计算世界坐标（反投影）
    # world_coords = []
    # for i in range(image_points_homogeneous.shape[1]):
    #     uv1 = image_points_homogeneous[:, i]  # 3x1
    #     cam_coords = K_inv @ uv1  # 归一化相机坐标系中的点
    #     world_coord = np.linalg.inv(R) @ (cam_coords - translation_vector.flatten())  # 计算世界坐标
    #     world_coords.append(world_coord)
    # 计算投影矩阵的伪逆
    P_inv = np.linalg.pinv(projection_matrix.T).T  # (3, 4)

    # 计算世界坐标 (N, 4)
    world_coords_homogeneous = (P_inv @ image_points_homogeneous).T  # (N, 4)

    # 归一化世界坐标（确保第四列为1）
    world_coords = world_coords_homogeneous[:, :3] / world_coords_homogeneous[:, 3][:, np.newaxis]

    # 固定 Z = depth
    scale = depth / world_coords[:, 2]
    world_coords[:, :2] *= scale[:, np.newaxis]
    world_coords[:, 2] = depth  # 设定 Z 轴深度

    return np.array(world_coords)  # (N, 3)

# 示例数据
camera_intrinsic=np.array([[2.51112220e+03, 0.00000000e+00, 9.71406297e+02],
 [0.00000000e+00, 2.51206946e+03, 5.73149627e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
image_points = np.array([[1083, 853],[1113,682.5]], dtype=np.float32)  # 假设是像素坐标
dist_coeffs = np.array([[-8.37716727e-01, 1.84486584e+00, 3.00158084e-03, 5.51622873e-03, -3.21176567e+00]], dtype=np.float32)
projection_matrix=np.array([[ 2.63503038e+03 ,6.80691108e+01, -5.48950268e+02,  1.25617746e+04],
[-3.22203165e+02,  3.81418912e+01, -2.55611498e+03,  8.91014097e+03],
 [ 2.93252716e-01,  9.23466343e-01, -2.47412121e-01,  1.03355228e+01]], dtype=np.float32)

# 转换
utm_coords = image_to_utm(image_points, camera_intrinsic, projection_matrix, dist_coeffs)
print(utm_coords)  # 形状为 (N, 3)