import copy
from typing import Tuple, List
import numpy as np

class IRPattern(object):
    def __init__(self, image_shape: Tuple[int, int], camera_intr: np.array):
        """Create an xy coordinate matrix of the camera image.

        Args:
            image_shape (Tuple[int, int]): (2) shape of image to add ir patterns
            camera_intr (np.array): (3, 3) camera intrinsic matrix.
        """
        cols = np.arange(image_shape[1])
        rows = np.arange(image_shape[0])

        self.image_shape = image_shape

        self.coords = np.empty((len(rows), len(cols), 2), dtype=np.float32)
        self.coords[:,:,0] = rows[:, None]
        self.coords[:,:,1] = cols

        self.camera_intr = camera_intr
        self.fx = camera_intr[0][0]
        self.fy = camera_intr[1][1]
        self.cx = camera_intr[0][2]
        self.cy = camera_intr[1][2]

    def gen_simple_ir_pattern(self, dot_length: float=22, dot_diameter: float=4, rotation: List[float]=[20, 37])-> np.array:        
        # The diameter of the dot when the dot length becomes unit length.
        _unit_dot_size = dot_diameter / dot_length
        # A matrix that sets the pixels of the depth image with the it pattern to 'True'. Output of this function.
        _matrix = np.zeros(self.image_shape, dtype=bool)
        # Copy xy coordinate matrix from self.coords in init function.
        _img = copy.deepcopy(self.coords)

        for _rad in np.deg2rad(rotation):
            _R = np.array([[np.cos(_rad), np.sin(_rad)], [-np.sin(_rad),  np.cos(_rad)]])

            _rot_img = (_img[:,:,]/ dot_length).dot(_R)
            
            _rot_img[:,:,] -= np.round(_rot_img[:,:,])
                
            _rot_img = np.linalg.norm(_rot_img, axis=2)
    
            _matrix[np.where(_rot_img < _unit_dot_size)] = True

        return _matrix

    def ir_matrix_from_depth(self, depth_image: np.array, transpose: float, dot_length: float=10.5, dot_diameter: float=2, rotation: List[float]=[50, 60])-> np.array:
        """_summary_

        Args:
            depth_image (np.array): (H, W) depth image to check ir pattern.
            transpose (float): A value for how far the camera is from the ir pattern projector in hardware. The IR projector and camera are always spaced apart in the width direction.
            dot_length (float, optional): Dot spacing in ir pattern. Defaults to 22.
            dot_size (float, optional): Dot size in ir pattern. Defaults to 4.
            rotation (List[float], optional): List of tilt angles for ir patterns. Defaults to [20, 37].

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

        # 
        _img[:,:,0] = (_img[:,:,0] - self.cy) * np.abs(depth_image) / self.fy
        _img[:,:,1] = (_img[:,:,1] - self.cx) * np.abs(depth_image) / self.fx

        _img[:,:,1] -= transpose

        _img_radius = np.linalg.norm(np.concatenate((_img,depth_image.reshape(self.image_shape[0], self.image_shape[1], 1)),axis=2), axis=2)

        _img[:,:,0] /= _img_radius
        _img[:,:,1] /= _img_radius

        _ir_img = np.zeros(_img.shape)
        _ir_img[:,:,0] = _img[:,:,0] * self.fy + self.cy
        _ir_img[:,:,1] = _img[:,:,1] * self.fx + self.cx

        for _rad in np.deg2rad(rotation):
            _R = np.array([[np.cos(_rad), np.sin(_rad)], [-np.sin(_rad),  np.cos(_rad)]])

            _rot_img = (_ir_img[:,:,]/ dot_length).dot(_R)
            
            _rot_img[:,:,] -= np.round(_rot_img[:,:,])
                
            _rot_img = np.linalg.norm(_rot_img, axis=2)
    
            _matrix[np.where(_rot_img < _unit_dot_size)] = True

        return _matrix
    def ir_matrix_from_depth_plane(self, depth_image: np.array, transpose: float, dot_length: float=10.5, dot_diameter: float=2, rotation: List[float]=[50, 60])-> np.array:
        """_summary_

        Args:
            depth_image (np.array): (H, W) depth image to check ir pattern.
            transpose (float): A value for how far the camera is from the ir pattern projector in hardware. The IR projector and camera are always spaced apart in the width direction.
            dot_length (float, optional): Dot spacing in ir pattern. Defaults to 22.
            dot_size (float, optional): Dot size in ir pattern. Defaults to 4.
            rotation (List[float], optional): List of tilt angles for ir patterns. Defaults to [20, 37].

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

        # 
        _img[:,:,0] = (_img[:,:,0] - self.cx) * np.abs(depth_image) / self.fx
        _img[:,:,1] = (_img[:,:,1] - self.cy) * np.abs(depth_image) / self.fy
        
        _img[:,:,1] -= transpose
        _img[:,:,0] /= np.abs(depth_image)
        _img[:,:,1] /= np.abs(depth_image)

        _ir_img = np.zeros(_img.shape)
        _ir_img[:,:,0] = _img[:,:,0] * self.fx + self.cx
        
        _ir_img[:,:,1] = _img[:,:,1] * self.fy + self.cy

        for _rad in np.deg2rad(rotation):
            _R = np.array([[np.cos(_rad), np.sin(_rad)], [-np.sin(_rad),  np.cos(_rad)]])

            _rot_img = (_ir_img[:,:,]/ dot_length).dot(_R)
            
            _rot_img[:,:,] -= np.round(_rot_img[:,:,])
                
            _rot_img = np.linalg.norm(_rot_img, axis=2)
    
            _matrix[np.where(_rot_img < _unit_dot_size)] = True

        return _matrix

    def segmask_transpose(self, depth_image: np.array, segmask: np.array, seg_intrinsic: np.array, transpose: float, scale: float):
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
        x_max, x_min = np.max(_seg[:,0])*scale, np.min(_seg[:,0]/scale)
        y_max, y_min = np.max(_seg[:,1])*scale, np.min(_seg[:,1]/scale)

        _img[:,:,0] = (_img[:,:,0] - self.cx) * np.abs(depth_image) / self.fx
        _img[:,:,1] = (_img[:,:,1] - self.cy) * np.abs(depth_image) / self.fy

        _img[:,:,1] += transpose

        _ir_img = np.zeros(_img.shape)
        _ir_img[:,:,0] = ((_img[:,:,0] * seg_intrinsic[0][0]) / np.abs(depth_image) + seg_intrinsic[0][2]).astype(int)
        _ir_img[:,:,1] = ((_img[:,:,1] * seg_intrinsic[1][1]) / np.abs(depth_image) + seg_intrinsic[1][2]).astype(int)

        true_list = np.where((_ir_img[:,:,0]<x_max) & (_ir_img[:,:,0]>x_min) & (_ir_img[:,:,1]<y_max) & (_ir_img[:,:,1]>y_min))

        _matrix[true_list] = True

        return _matrix
