import copy
import cv2
from typing import Tuple, List
import numpy as np

class IRPattern(object):
    def __init__(self, image_shape: Tuple[int, int], camera_intrinsic: np.array):
        """Create an xy coordinate matrix of the camera image.

        Args:
            image_shape (Tuple[int, int]): (2) shape of image to add ir patterns
            camera_intrinsic (np.array): (3, 3) camera intrinsic matrix.
        """
        # Save the shape of the image.
        cols = np.arange(image_shape[1])
        rows = np.arange(image_shape[0])
        self.image_shape = image_shape

        # Create a coordinate matrix with the pixel positions of the image as x and y values.
        self.coords = np.empty((len(rows), len(cols), 2), dtype=np.float32)
        self.coords[:,:,0] = rows[:, None]
        self.coords[:,:,1] = cols

        # Save the camera intrinsic data.
        self.fx = camera_intrinsic[0][0]
        self.fy = camera_intrinsic[1][1]
        self.cx = camera_intrinsic[0][2]
        self.cy = camera_intrinsic[1][2]

    def gen_simple_ir_pattern(self, transpose: float=0, dot_length: float=12.6, dot_diameter: float=3, rotation: List[float]=[50, 60])-> np.array:        
        # Generate IR pattern matrix from example depth image with a depth of 1.
        return self.ir_matrix_from_depth(np.ones(self.image_shape), transpose, dot_length, dot_diameter, rotation)

    def ir_matrix_from_depth(self, depth_image: np.array, transpose: float, dot_length: float=12.6, dot_diameter: float=3, rotation: List[float]=[50, 60])-> np.array:
        """A function that creates an IR pattern matrix using the values ​​of the depth image. A matrix of the same size as the Input Depth image is created, recording whether an IR pattern exists at the corresponding pixel location.

        Args:
            depth_image (np.array): (H, W) depth image to check IR pattern.
            transpose (float): A value for how far the camera is from the IR pattern projector in hardware. The IR projector and camera are always spaced apart in the width direction.
            dot_length (float, optional): Dot spacing in IR pattern. Defaults to 12.6.
            dot_size (float, optional): Dot size in IR pattern. Defaults to 3.
            rotation (List[float], optional): List of tilt angles for IR patterns. Defaults to [50, 60].

        Returns:
            np.array: (H, W) boolean type matrix. Pixels with an IR pattern are True.
        """
        # Check whether shape of depth image and image from initial function is same.
        if depth_image.shape != self.coords.shape[:2]:
            raise Exception("The shape of depth image and image in initial are mismatched")
        
        # The diameter of the dot when the dot length becomes unit length.
        _unit_dot_size = dot_diameter / dot_length
        # A matrix that sets the pixels of the depth image with the it pattern to 'True'. Output of this function.
        _matrix = np.zeros(depth_image.shape, dtype=bool)
        # Copy xy coordinate matrix from self.coords in init function.
        _img = copy.deepcopy(self.coords)

        # Convert each depth points in image into camera coordinate.
        _img[:,:,0] = (_img[:,:,0] - self.cy) * np.abs(depth_image) / self.fy
        _img[:,:,1] = (_img[:,:,1] - self.cx) * np.abs(depth_image) / self.fx

        # Transpose each depth points to IR projector coordinate.
        _img[:,:,1] -= transpose

        # Distance from each points and IR projector.
        _img_radius = np.linalg.norm(np.concatenate((_img,depth_image.reshape(self.image_shape[0], self.image_shape[1], 1)),axis=2), axis=2)

        # Pattern calculation when each point is at a distance of 1 from the IR projector.
        _img[:,:,0] /= _img_radius
        _img[:,:,1] /= _img_radius

        # Convert points to IR projector image coordinate.
        _img[:,:,0] = _img[:,:,0] * self.fy + self.cy
        _img[:,:,1] = _img[:,:,1] * self.fx + self.cx

        # Calculates patterns according to all angles.
        for _rad in np.deg2rad(rotation):
            # Rotation matrix
            _R = np.array([[np.cos(_rad), np.sin(_rad)], [-np.sin(_rad),  np.cos(_rad)]])

            # For convenience, the coordinates of the points are divided by the unit length and all points are rotated.
            _rot_img = (_img[:,:,]/ dot_length).dot(_R)
            
            # Move each point to find the distance to the nearest vertex.
            _rot_img[:,:,] -= np.round(_rot_img[:,:,])
                
            # Calculate distance from origin
            _rot_img = np.linalg.norm(_rot_img, axis=2)
    
            # True if smaller than IR pattern dot size
            _matrix[np.where(_rot_img < _unit_dot_size)] = True

        # Output is a matrix in which pixels with IR patterns are True.
        return _matrix
    def ir_matrix_from_depth_plane(self, depth_image: np.array, transpose: float, dot_length: float=12.6, dot_diameter: float=3, rotation: List[float]=[50, 60])-> np.array:
        """A function that creates an IR pattern matrix using the values ​​of the depth image. A matrix of the same size as the Input Depth image is created, recording whether an IR pattern exists at the corresponding pixel location.
        Unlike function ir_matrix_from_depth, an IR pattern without distortion is generated. 

        Args:
            depth_image (np.array): (H, W) depth image to check IR pattern.
            transpose (float): A value for how far the camera is from the IR pattern projector in hardware. The IR projector and camera are always spaced apart in the width direction.
            dot_length (float, optional): Dot spacing in IR pattern. Defaults to 12.6.
            dot_size (float, optional): Dot size in IR pattern. Defaults to 3.
            rotation (List[float], optional): List of tilt angles for IR patterns. Defaults to [50, 60].

        Returns:
            np.array: (H, W) boolean type matrix. Pixels with an ir pattern are True.
        """
        # Check whether shape of depth image and image from initial function is same.
        if depth_image.shape != self.coords.shape[:2]:
            raise Exception("The shape of depth image and image in initial are mismatched")
        
        # The diameter of the dot when the dot length becomes unit length.
        _unit_dot_size = dot_diameter / dot_length
        # A matrix that sets the pixels of the depth image with the it pattern to 'True'. Output of this function.
        _matrix = np.zeros(depth_image.shape, dtype=bool)
        # Copy xy coordinate matrix from self.coords in init function.
        _img = copy.deepcopy(self.coords)

        # Convert each depth points in image into camera coordinate.
        _img[:,:,0] = (_img[:,:,0] - self.cy) * np.abs(depth_image) / self.fy
        _img[:,:,1] = (_img[:,:,1] - self.cx) * np.abs(depth_image) / self.fx
        
        # Transpose each depth points to IR projector coordinate.
        _img[:,:,1] -= transpose
        # Move all points to a position with depth 1.
        _img[:,:,0] /= np.abs(depth_image)
        _img[:,:,1] /= np.abs(depth_image)

        # Convert points to IR projector image coordinate.
        _img[:,:,0] = _img[:,:,0] * self.fy + self.cy
        _img[:,:,1] = _img[:,:,1] * self.fx + self.cx

        # Calculates patterns according to all angles.
        for _rad in np.deg2rad(rotation):
            # Rotation matrix
            _R = np.array([[np.cos(_rad), np.sin(_rad)], [-np.sin(_rad),  np.cos(_rad)]])

            # For convenience, the coordinates of the points are divided by the unit length and all points are rotated.
            _rot_img = (_img[:,:,]/ dot_length).dot(_R)
            
            # Move each point to find the distance to the nearest vertex.
            _rot_img[:,:,] -= np.round(_rot_img[:,:,])
                
            # Calculate distance from origin
            _rot_img = np.linalg.norm(_rot_img, axis=2)
    
            # True if smaller than IR pattern dot size
            _matrix[np.where(_rot_img < _unit_dot_size)] = True

        # Output is a matrix in which pixels with IR patterns are True.
        return _matrix

    def segmask_transpose(self, depth_image: np.array, segmask: np.array, seg_intrinsic: np.array, transpose: float=0, scale: float=1.05):
        """TODO: complete this function

        Args:
            depth_image (np.array): _description_
            segmask (np.array): _description_
            seg_intrinsic (np.array): _description_
            transpose (float): _description_
            scale (float): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """

        print("TODO: segmask_transpose in IRPattern Class not complete")

        if depth_image.shape != segmask.shape:
            raise Exception("The shape of depth image and segmask are mismatched")
        
        # depth_image[np.where(depth_image==0)]+=0.00001

        _matrix = np.zeros(depth_image.shape, dtype=bool)
        _img = copy.deepcopy(self.coords)
        _seg = copy.deepcopy(self.coords)
        _seg = _seg[np.where(segmask)]
        y_max, y_min = np.max(_seg[:,0])*scale, np.min(_seg[:,0]/scale)
        x_max, x_min = np.max(_seg[:,1])*scale, np.min(_seg[:,1]/scale)

        _img[:,:,0] = (_img[:,:,0] - self.cy) * np.abs(depth_image) / self.fy
        _img[:,:,1] = (_img[:,:,1] - self.cx) * np.abs(depth_image) / self.fx
        

        _img[:,:,0] += transpose

        _ir_img = np.zeros(_img.shape)
        _ir_img[:,:,0] = ((_img[:,:,0] * seg_intrinsic[1][1]) / np.abs(depth_image) + seg_intrinsic[1][2]).astype(int)
        _ir_img[:,:,1] = ((_img[:,:,1] * seg_intrinsic[0][0]) / np.abs(depth_image) + seg_intrinsic[0][2]).astype(int)

        true_list = np.where((_ir_img[:,:,0]<y_max) & (_ir_img[:,:,0]>y_min) & (_ir_img[:,:,1]<x_max) & (_ir_img[:,:,1]>x_min))

        _matrix[true_list] = True

        return _matrix
    
    @staticmethod
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
