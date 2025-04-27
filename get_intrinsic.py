import cv2
import numpy as np
import glob
import os


def calibrate_camera_from_images(row, col, image_dir):
    """
    进行相机标定，计算相机矩阵和畸变系数

    :param row: 棋盘格行数
    :param col: 棋盘格列数
    :param image_dir: 棋盘格图片文件夹路径
    :return: (camera_matrix, dist_coeffs) 相机矩阵和畸变系数
    """
    # 每个棋盘格的边长 (单位: 米)
    square_size = 24.5e-3  # 24.5mm 转换为米

    # 生成棋盘格的 3D 角点坐标
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2) * square_size

    # 存储 3D 角点坐标 和 2D 角点坐标
    objpoints = []  # 3D 真实世界坐标
    imgpoints = []  # 2D 图像坐标

    # 获取所有棋盘格图片路径
    image_paths = glob.glob(os.path.join(image_dir, "chessboard*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        success, corners = cv2.findChessboardCorners(gray, (row, col), None)

        if success:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
            )
            imgpoints.append(corners_refined)
            # 绘制棋盘格角点并显示
            # cv2.drawChessboardCorners(img, (row, col), corners_refined, success)
            # cv2.imshow("corners", img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

    # 读取第一张图片获取图像尺寸
    img = cv2.imread(image_paths[10])
    img_shape = (img.shape[1], img.shape[0])

    # 进行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    print('\nRMS: ',ret);

    # 对图片进行去畸变
    # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    # cv2.imshow("Undistorted Image", undistorted_img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    return camera_matrix, dist_coeffs


# camera_matrix, dist_coeffs = calibrate_camera_from_images(6, 8, "chessboard_image")
# print("Camera Matrix:\n", camera_matrix)
# print("\nDistortion Coefficients:\n", dist_coeffs)
