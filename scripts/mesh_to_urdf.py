import os
import glob
import argparse
from itertools import repeat
import xml.etree.cElementTree as ET
from multiprocessing import Pool
import numpy as np
import trimesh

# mesh rescale prameters
GRIPPER_WIDTH = 0.1
DENSE_PET   = 1310  # [kg/m^3]
DENSE_WATER = 997   # [kg/m^3]
DENSE_PAPER = 0.075 # [kg/m^3]
EXT = ['.obj', '.stl', '.STL']

# Parse arguments
parser = argparse.ArgumentParser(description='This script converts mesh data to Isaac Gym asset.')
parser.add_argument('--mesh', required=True, help='Path to mesh folder')
parser.add_argument('--urdf', required=True, help='Path to urdf folder')
args = parser.parse_args()

mesh_root_dir = args.mesh
urdf_root_dir = args.urdf

if not os.path.exists(mesh_root_dir):
    print('Invalid Mesh Directory')
    exit()
if not os.path.exists(urdf_root_dir):
    os.makedirs(urdf_root_dir)

def indent(elem, level=0, more_sibs=False):
    ''' Add indent when making URDF file'''
    # https://stackoverflow.com/questions/749796/pretty-printing-xml-in-python
    i = "\n"
    if level:
        i += (level-1) * '  '
    num_kids = len(elem)
    if num_kids:
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
            if level:
                elem.text += '  '
        count = 0
        for kid in elem:
            indent(kid, level+1, count < num_kids - 1)
            count += 1
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            if more_sibs:
                elem.tail += '  '
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            if more_sibs:
                elem.tail += '  '


def obj_to_urdf(mesh_dir):
    for ext in EXT:
        obj_files = glob.glob(os.path.join(mesh_root_dir, '*' + ext))
        print('Mesh num [', ext, ']:', len(obj_files), '\n')
        _process_count = 1
        for mesh_file in obj_files:
            # Get name of target mesh
            mesh_name = os.path.basename(mesh_file).split('.')[0]
            print('Processing (' , _process_count, '/' , len(obj_files), '): [ ' + mesh_name + ' ]')
            _process_count += 1
            # Load mesh
            mesh = trimesh.load(os.path.join(mesh_file))
            mesh_tri = trimesh.load(os.path.join(mesh_file), process=False)
            # Rescale mesh
            exts = mesh_tri.bounding_box_oriented.primitive.extents
            max_dim = np.max(exts)
            scale = GRIPPER_WIDTH / max_dim
            mesh.apply_scale(scale) # mm to m scale
            mesh_tri.apply_scale(scale) # mm to m scale

            trimesh.repair.broken_faces(mesh_tri)
            trimesh.repair.fix_inversion(mesh_tri, multibody=True)
    
            mesh.density = DENSE_PET
            mesh_tri.density = DENSE_PET
            mesh.vertices -= mesh_tri.center_mass
            mesh_tri.vertices -= mesh_tri.center_mass

            # save mesh
            if os.path.exists(os.path.join(urdf_root_dir, mesh_name)):
                print('\toveride existing file for: [ ', mesh_name, ' ]')
            else:
                os.makedirs(urdf_root_dir + '/' + mesh_name)
            mesh.export(os.path.join(urdf_root_dir, mesh_name, mesh_name + ext))

            # create urdf file
            urdf = ET.Element('robot', name=mesh_name)
            link = ET.SubElement(urdf, 'link', name=mesh_name)
            inertial = ET.SubElement(link, 'inertial')
            mass = ET.SubElement(inertial, 'mass', value=str(mesh_tri.mass))
            inertia_dict = {'ixx': str(mesh_tri.moment_inertia[0, 0]),
                            'ixy': str(mesh_tri.moment_inertia[0, 1]),
                            'ixz': str(mesh_tri.moment_inertia[0, 2]),
                            'iyy': str(mesh_tri.moment_inertia[1, 1]),
                            'iyz': str(mesh_tri.moment_inertia[1, 2]),
                            'izz': str(mesh_tri.moment_inertia[2, 2])}
            inertia = ET.SubElement(inertial, 'inertia', inertia_dict)

            visual = ET.SubElement(link, 'visual')
            origin = ET.SubElement(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
            geometry = ET.SubElement(visual, 'geometry')
            _mesh = ET.SubElement(geometry, 'mesh', filename=os.path.join(urdf_root_dir, mesh_name, mesh_name + ext), scale='1 1 1')

            collision = ET.SubElement(link, 'collision')
            origin = ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
            geometry = ET.SubElement(collision, 'geometry')
            _mesh = ET.SubElement(geometry, 'mesh', filename=os.path.join(urdf_root_dir, mesh_name, mesh_name + ext), scale='1 1 1')

            # save urdf file
            indent(urdf)
            tree = ET.ElementTree(urdf)
            with open(os.path.join(urdf_root_dir, mesh_name, mesh_name + '.urdf'), 'wb') as f:
                tree.write(f, encoding='utf-8', xml_declaration=True)
            mesh_tri.apply_scale(100) # to find better stable pose (bigger, the better find stable pose)

            # get stable poses
            stable_poses, prob = mesh_tri.compute_stable_poses(center_mass=mesh_tri.center_mass,n_samples=10, sigma=0.1)
            for i in range(len(stable_poses)):
                stable_poses[i][0:3, 3] *= 0.001

            np.save(os.path.join(urdf_root_dir, mesh_name, 'stable_poses.npy'), stable_poses)
            np.save(os.path.join(urdf_root_dir, mesh_name, 'stable_prob.npy'), prob)

            # save log as txt
            with open(os.path.join(urdf_root_dir, mesh_name, 'log.txt'), 'w') as f:
                f.write('num stable poses: {}\n'.format(len(stable_poses)))
                s = 'prob: '
                for i, p in enumerate(prob):
                    s += '{:.3f}'.format(p)
                    if i < len(prob) - 1:
                        s += ', '
                s += '\n'
                f.write(s)
            print('\tfinish [ ', mesh_name, ']\n')


if __name__ == '__main__':
    # get file directory
    obj_to_urdf(mesh_root_dir)