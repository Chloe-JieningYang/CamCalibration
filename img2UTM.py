import numpy as np
import cv2

def image_to_utm(image_points: np.ndarray, camera_intrinsic: np.ndarray, R: np.ndarray, T: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """
    将像素坐标转换为 UTM 坐标。

    参数：
        image_points (np.ndarray): 形状 (N, 2) 的像素坐标数组
        camera_intrinsic (np.ndarray): 相机内参矩阵 (3x3)
        dist_coeffs (np.ndarray): 畸变系数 (1x5 或 None)
        R (np.ndarray): 旋转矩阵 (3x3)
        T (np.ndarray): 平移向量 (3x1)

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

    # 计算相机内参的逆矩阵
    K_inv = np.linalg.inv(camera_intrinsic)
    R_inv = np.linalg.inv(R)
    
    world_coords = []
    for i in range(image_points_homogeneous.shape[1]):
        # 像素坐标转相机坐标
        uv1 = image_points_homogeneous[:, i]  # 3x1
        cam_coords = K_inv @ uv1  # 归一化相机坐标系中的点
        
        # 相机坐标转世界坐标
        world_coord = R_inv @ (cam_coords - T.flatten())  # 计算世界坐标
        
        # 缩放到z=1平面
        scale = 1.0 / world_coord[2] if world_coord[2] != 0 else 1.0
        world_coord *= scale
        
        world_coords.append(world_coord)
    
    return np.array(world_coords)  # (N, 3)

# 示例数据
camera_intrinsic=np.array([[2.51112220e+03, 0.00000000e+00, 9.71406297e+02],
 [0.00000000e+00, 2.51206946e+03, 5.73149627e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
image_points = np.array([[1083, 853],[1113,682.5]], dtype=np.float32)  # 假设是像素坐标
dist_coeffs = np.array([[-8.37716727e-01, 1.84486584e+00, 3.00158084e-03, 5.51622873e-03, -3.21176567e+00]], dtype=np.float32)
rotation_matrix=np.array([[ 0.85439444, -0.50240244, -0.13267227],
                        [-0.18674612, -0.0586158,  -0.98065798,],
                        [ 0.48490827,  0.86264476, -0.14390272]], dtype=np.float32)
translation_vector=np.array([[-0.63074433],[-0.44910791],[10.59422531]],dtype=np.float32)

# 转换
utm_coords = image_to_utm(image_points, camera_intrinsic, rotation_matrix, translation_vector, dist_coeffs)
print(utm_coords)  # 形状为 (N, 3)