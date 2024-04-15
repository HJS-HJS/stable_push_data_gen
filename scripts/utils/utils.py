import numpy as np
import cv2

def add_depth_noise(depth_image: np.array, noise_type: str="gauss_field", scale: float=0.0012, var: float=0.000001) -> np.array:
    """add noise in depth image

    Args:
        depth_image (np.array): (H, W) depth image to add noise
        noise_type (str, optional): Kind of filter. Defaults to "gauss_field".
        scale (float, optional): scale param for gauss field filter. Defaults to 0.1.
        var (float, optional): varidation param for gauss filter. Defaults to 0.000001.

    Returns:
        np.array: (H, W) with filterd depth image.
    """
    
    # When 'noise_type' is gauss field
    if noise_type == "gauss_field":
        # get depth image shape
        row,col= depth_image.shape
        # Generate noise filter
        gp_noise = np.random.normal(scale=scale,size=(100, 100))
        gp_noise = cv2.resize(gp_noise,(col,row))
        gp_noise = gp_noise.astype(np.float16)
        return depth_image + gp_noise
    
    # When 'noise_type' is gauss
    elif noise_type == "gauss":
        # get depth image shape
        row,col= depth_image.shape
        # Set parameters for noise filter
        mean = 0
        var = 0.000001
        sigma = var**0.5
        # Generate noise filter
        gp_noise = np.random.normal(mean,sigma,size=(row,col))
        gp_noise = gp_noise.reshape(row,col)
        return depth_image + gp_noise
    
    # When 'noise_type' is invalid
    else:
        print("[ Depth Noise Filter ]: Invalide noise type")
        return depth_image

def add_pcd_noise(pcd: np.array, noise_type: str="gauss") -> np.array:
    """add noise in pointcloud data

    Args:
        pcd (np.array): Pointcloud data to add noise
        noise_type (str): Kind of filter (gauss, sin). Defaults to gauss.

    Returns:
        np.array: Pointcloud data array with noise
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

def depth_to_pcd(depth_image: np.array, camera_intr: np.array) ->np.array:
    """Convert depth image to pointcloud data.

    Args:
        depth_image (np.array): (H, W) depth image to convert.
        camera_intr (np.array): (3, 3) camera intrinsic matrix.

    Returns:
        np.array: (N, 3) pointcloud data array converted from depth image
    """
    height, width = depth_image.shape
    row_indices = np.arange(height)
    col_indices = np.arange(width)
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth_image.flatten(), [3, 1])
    point_cloud = depth_arr * np.linalg.inv(camera_intr).dot(pixels_homog)
    return point_cloud.transpose()

def img_to_depth(image1: np.array, image2: np.array, intrinsic: np.array, baseline: float=0.55, disparities: int=64, block: int=31, units: float=0.512) ->np.array:
    """Find a depth map from disparity map. StereoBM from cv is used to get disparity map. StereoBM is area-based local block matching algorithm using window.
    When implemented in hardware, only pixel information of a certain size is required, so not only does it require less memory than the global matching method, but the operation itself is simple, making it suitable for design in hardware.
    However, because it has relatively low accuracy, various post-processing algorithms can be used to increase accuracy or through aggregation methods.
    This function does not include any post-processing.

    Args:
        image1 (np.array): (H, W) with `CV_8UC1` gray image. This is the standard image when creating a depth image.
        image2 (np.array): (H, W) with `CV_8UC1` gray image. Image for which disparity is to be calculated. The size of the image must also be the same as image1.
        intrinsic (np.array): (3, 3) camera intrinsic matrix. Used when calculating the focal length of a lens.
        baseline (float, optional): Distance in [m] between the two cameras. Defaults to 55.
        disparities (int, optional): Num of disparities to consider. Defaults to 64.
        block (int, optional): Block size to match. Defaults to 31.
        units (float, optional): Depth units, adjusted for the output to fit in one byte. Defaults to 0.512.

    Returns:
        np.array: (H, W) with `float` created depth image. The depth image from the same viewpoint as image1 is the output.
    """
    # Set parameter
    fx = intrinsic[0][0]    # lense focal length

    # Initialize StereoBM
    sbm = cv2.StereoBM_create(numDisparities=disparities,
                          blockSize=block)

    # Create disparity map.
    # StereoBM use SAD algorithm (Sum of Absolute Difference)
    # The absolute value of the difference between the values ​​of pixels within the left and right windows is taken and added to calculate the matching cost.
    disparity = sbm.compute(image1, image2)

    # Create depth map from disparity map
    valid_pixels = disparity > 0
    depth_img = np.zeros(shape=image1.shape).astype("float32")
    depth_img[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

    return np.array(depth_img, dtype = np.float32)
