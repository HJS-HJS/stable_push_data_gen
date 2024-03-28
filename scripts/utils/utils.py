import numpy as np
import matplotlib.pyplot as plt
import cv2

def add_depth_noise(noise_type: str, depth_image: np.array) -> np.array:
    """_summary_

    Args:
        noise_type (str): gauss, gauss_field
        depth_image (np.array): input depth image

    Returns:
        np.array: noisy image
    """

    if noise_type == "gauss":
        row,col= depth_image.shape
        mean = 0
        var = 0.000001
        sigma = var**0.5
        gp_noise = np.random.normal(mean,sigma,(row,col))
        gp_noise = gp_noise.reshape(row,col)
        return depth_image + gp_noise
    
    elif noise_type == "gauss_field":
        gp_noise = np.random.normal(scale=0.007, size=(100, 100))
        gp_noise = cv2.resize(gp_noise, (1960, 1220))
        gp_noise = gp_noise.astype(np.float32)
        return depth_image + gp_noise
    
    else:
        print("[ Depth Noise Filter ]: Invalide noise type")
        return depth_image

def add_pcd_noise(noise_type: str, pcd: np.array) -> np.array:
    """_summary_

    Args:
        noise_type (str): gauss, sin
        depth_image (np.array): input depth image

    Returns:
        np.array: noisy image
    """
    if noise_type == "gauss":
        row,col= pcd.shape
        mean = 0
        var = 0.00001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = pcd + gauss
        return pcd
    
    elif noise_type == "sin":
        pcd = np.transpose(pcd)
        _amplitude = 0.3 / 2
        _freqx, _freqy = 180 + 60 * (1 - 2 * np.random.random_sample(2))
        _offset = np.sin(pcd[0]*_freqx) * np.cos(pcd[1]*_freqy) * np.power(_amplitude * (1 - _amplitude/2), 2)
        _offset *= 0.4 * np.sin(np.linalg.norm([pcd[0], pcd[1]], 2) * _freqx * 1.3) + 1
        pcd[2] += _offset
        noisy = np.transpose(pcd)
        return noisy

    else:
        print("[ PCD Noise Filter ]: Invalide noise type")
        return pcd

def depth_to_pcd(depth_image, camera_intr):
    height, width = depth_image.shape
    row_indices = np.arange(height)
    col_indices = np.arange(width)
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth_image.flatten(), [3, 1])
    point_cloud = depth_arr * np.linalg.inv(camera_intr).dot(pixels_homog)
    return point_cloud.transpose()
