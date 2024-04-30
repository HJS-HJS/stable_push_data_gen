import os
import copy
import numpy as np
import open3d as o3d

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_directory, "../assets/urdf/")

# Define arguments to pass to the Python script
objects = os.listdir(assets_dir)
objects.sort()

class SelectStablePose(object):
    
    def __init__(self, asset_dir):
        
        self.is_modified = None
        self.current_idx = 0
        self.current_pose = None
        self.asset_dir = asset_dir
        self.objects = os.listdir(asset_dir)
        self.objects.sort()

        self.check_for_all_objects()
        
    def check_for_all_objects(self):
        a = 0
        for object in self.objects:
            self.current_object = object
            self.check_for_object()
            # if a % 11 == 0:
                # self.check_for_object()
            # a += 1

    def check_for_object(self):
        self.is_modified = False
        self.stable_poses = np.load(self.asset_dir + self.current_object + '/stable_poses.npy', allow_pickle=True)
        self.stable_probs = np.load(self.asset_dir + self.current_object + '/stable_prob.npy', allow_pickle=True)
        # Normal case where there are multiple stable poses
        print("Pose: ", len(self.stable_poses.shape), "\tCurrent object: ", self.current_object)
        if len(self.stable_poses.shape) == 3:
            idx = 0
            while (idx<self.stable_poses.shape[0]):
                if self.is_modified:
                    break
                print("Idx: ", idx, "/", self.stable_poses.shape[0]-1, "\tProb: ", self.stable_probs[idx])
                self.current_idx = idx
                self.current_pose = self.stable_poses[idx]
                idx+=self.visualize_in_given_pose()
                if idx<0 :idx+=1

        # Case where there is only one stable pose        
        if len(self.stable_poses.shape) == 2:
            pass
            self.current_pose = self.stable_poses
            self.visualize_in_given_pose()
                
    def visualize_in_given_pose(self) :
        
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(self.asset_dir + self.current_object + '/' + self.current_object + '.obj')
        if mesh.is_empty():
            mesh = o3d.io.read_triangle_mesh(self.asset_dir + self.current_object + '/' + self.current_object + '.stl')
        if mesh.is_empty():
            mesh = o3d.io.read_triangle_mesh(self.asset_dir + self.current_object + '/' + self.current_object + '.STL')

        mesh1 = copy.deepcopy(mesh).transform(self.current_pose)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
        
        # Load the text to visualize
        text_object = 'Object: ' + self.current_object 
        text_index = 'Pose index: ' + str(self.current_idx)
        
        text_object_pcd = self.text_3d(text_object, pos=[0, -0.05, 0.07], direction=[1, 0, 0], degree=0.0, density = 20, font='DejaVu Sans Mono for Powerline.ttf', font_size=2)
        text_index_pcd = self.text_3d(text_index, pos=[0, -0.05, 0.06], direction=[1, 0, 0], degree=0.0, density = 20, font='DejaVu Sans Mono for Powerline.ttf', font_size=2)
        
        
        text_notice = 'Press S to save the pose, C to close the window, E to exit the program'
        text_notice_pcd = self.text_3d(text_notice, pos=[0, -0.1, -0.07], direction=[1, 0, 0], degree=0.0, density = 20, font='DejaVu Sans Mono for Powerline.ttf', font_size=2)
        
        
        # Create the visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        
        self.next_idx = 1

        # Register the callback functions
        vis.register_key_callback(ord('S'), self.select_stable_pose)
        vis.register_key_callback(ord('C'), self.close_window)
        vis.register_key_callback(ord('E'), self.del_window)
        vis.register_key_callback(ord('B'), self.back_to_prev_idx)
        vis.register_key_callback(ord('.'), self.rotate_mesh_right)
        vis.register_key_callback(ord(','), self.rotate_mesh_left)
        vis.register_key_callback(ord('R'), self.reset_camera_pose)
        # vis.register_key_callback(ord('M'), self.modify_stable_pose)
        vis.create_window(window_name=self.current_object)

        # Add the geometry to the visualizer
        mesh1.compute_vertex_normals()
        vis.add_geometry(mesh1)
        vis.add_geometry(mesh_frame)
        
        # Add the text to the visualizer
        vis.add_geometry(text_object_pcd)
        vis.add_geometry(text_index_pcd)
        vis.add_geometry(text_notice_pcd)
        
        view_ctl = vis.get_view_control()
        view_ctl.set_front([1.0, 0.0, 0.0])
        view_ctl.set_up([0.0, 0.0, 1.0])
        # Run the visualizer
        vis.run()
        return self.next_idx
    
    @staticmethod
    def text_3d(text, pos, direction=None, degree=0.0, density = 10, font='/usr/share/fonts/truetype/ttf-bitstream-vera/VeraMoBd.ttf', font_size=16):
        # https://github.com/isl-org/Open3D/issues/2#issuecomment-610683341
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """
        if direction is None:
            direction = (0., 0., 1.)

        from PIL import Image, ImageFont, ImageDraw
        from pyquaternion import Quaternion

        try:
            font_obj = ImageFont.truetype(font, font_size*density)
        except IOError:
            font_obj = ImageFont.load_default()
            
        font_dim = font_obj.getsize(text)

        img = Image.new('RGB', font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 100.0 / density)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        
        if np.linalg.norm(raxis) < 1e-6:
            raxis = (0.0, 0.0, 1.0)
            
        trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
                Quaternion(axis=direction, degrees=degree)).transformation_matrix
        
        trans[0:3, 3] = np.asarray(pos)
        
        pcd.transform(trans)
        
        return pcd
  
    ## Callback functions
    def select_stable_pose(self, vis):
        # print(self.current_pose)
        with open(self.asset_dir + self.current_object + '/stable_poses.npy', 'wb') as f:
            np.save(f, self.current_pose)
        vis.close()
        print("Modified stable pose for object: ", self.current_object)
        print("Idx: ", self.current_idx)
        self.is_modified = True

    def close_window(self, vis):
        vis.close()

    def del_window(self, vis):
        vis.close()
        exit()
        
    def back_to_prev_idx(self, vis):
        self.next_idx = -1
        vis.close()

    def modify_stable_pose(self, vis):
        print(self.current_pose)
        print(type(self.current_pose))
        vis.close()
        r = np.identity(4)
        self.current_pose = self.current_pose.dot(r)
        print(r)
        print(type(r))
        print("Modified stable pose for object: ", self.current_object)
        print("Idx: ", self.current_idx)
        # self.visualize_in_given_pose()

    def rotate_mesh_right(self, vis):
        ctr = vis.get_view_control()
        ctr.rotate(-90.0, 0.0)

    def rotate_mesh_left(self, vis):
        ctr = vis.get_view_control()
        ctr.rotate(90.0, 0.0)

    def reset_camera_pose(self, vis):
        view_ctl = vis.get_view_control()
        view_ctl.set_front([1.0, 0.0, 0.0])
        view_ctl.set_up([0.0, 0.0, 1.0])

if __name__ == '__main__':
    select = SelectStablePose(assets_dir)
    