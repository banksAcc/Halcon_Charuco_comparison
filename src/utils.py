import cv2
import numpy as np
import yaml
import json

from scipy.spatial.transform import Rotation as R

def parse_args_from_json(config_path="settings.json"):
    """
    Reads parameters from a JSON file and returns an object with attributes.
    Equivalent to argparse.Namespace but from JSON.
    """
    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file: {e}")

    return Args(config)


def load_camera_calibration(yaml_path):
    """
    Reads OpenCV intrinsic and distortion parameters from YAML files.
    Returns camera_matrix (3×3) and dist_coeffs (1×8 vector), in pixel units.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    oc = data['IntrinsicCalibration']['OpenCV']
    fx, fy = oc['fx'], oc['fy']
    cx, cy = oc['cx'], oc['cy']
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)
    # ordiniamo k1…k6, p1, p2 in vettore dist_coeffs:
    dist_coeffs = np.array([
        oc['k1'], oc['k2'], oc['p1'], oc['p2'],
        oc['k3'], oc['k4'], oc['k5'], oc['k6']
    ], dtype=np.float64)
    return camera_matrix, dist_coeffs

def pose_to_matrix(rvec, tvec):
    """
    Converts a pose (rvec, tvec) to a 4x4 homogeneous transformation matrix.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3,)
    return T

def offset_pose_to_center(T_pose, offset_xyz):
    """
    Applies a local translation to the original frame to move the origin.
    offset_xyz is an array (3,) in meters, e.g. [0.0375, 0.0375, 0.0]
    """
    offset_local = np.eye(4)
    offset_local[:3, 3] = offset_xyz
    T_centered = T_pose @ offset_local
    return T_centered

def matrix_to_pose(T):
    """
    Extract rvec, tvec from a 4x4 transformation matrix
    """
    R_mat = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R_mat)
    return rvec, tvec

def rotation_matrix_to_quaternion(R_mat):
    """
    Converts a 3x3 rotation matrix to quaternion (qx, qy, qz, qw)
    """
    return R.from_matrix(R_mat).as_quat()  # [x, y, z, w]