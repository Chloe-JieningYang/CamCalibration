import numpy as np
import pandas as pd
from pyproj import Transformer

def get_utm_epsg(lon: float):
    # 计算 UTM 带号（UTM 带号计算公式）
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone}"


def read_gps_to_UTM(file_name: str) -> np.ndarray:
    """
    读取 GPS 数据文件，并返回UTM坐标的 NumPy 数组

    参数：
        file_name (str): GPS 数据 CSV 文件的路径

    返回：
        np.ndarray: 形状为 (N, 3) 的 NumPy 数组，包含Easting, Northing, 0
    """
    column_names = ["lat", "lon"]
    gps_data = pd.read_csv(file_name, header=None, usecols=[0, 1], names=column_names)

    # 创建 UTM 转换器（以第一行数据计算 UTM 带号）
    epsg_utm = get_utm_epsg(gps_data[0, 1])
    transformer = Transformer.from_crs("EPSG:4326", epsg_utm)

    # 初始化 N×3 矩阵
    utm_matrix = np.zeros((gps_data.shape[0], 3))
    # 转化成UTM
    utm_matrix[:, :2] = np.array(transformer.transform(gps_data[:, 0], gps_data[:, 1])).T
    return utm_matrix


def read_pixel_data(file_name: str) -> np.ndarray:
    """
    读取像素坐标数据文件，并返回像素坐标的 NumPy 数组。

    参数：
        file_name (str): 包含像素坐标的文本文件路径，每行包含逗号分隔的数值。

    返回：
        np.ndarray: 形状为 (N, 2) 的 NumPy 数组，包含像素坐标 (x, y)。
    """
    with open(file_name, "r") as file:
        pixel_data_list = [
            [float(value) for value in line.strip().split(",")] for line in file
        ]
    return np.array(pixel_data_list)[:, [0, 1]]

