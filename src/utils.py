# src/utils.py
import cv2
import numpy as np
import yaml

def load_camera_calibration(yaml_path):
    """
    Legge da file YAML i parametri intrinseci e di distorsione OpenCV.
    Restituisce camera_matrix (3×3) e dist_coeffs (vettore 1×8), in unità di pixel.
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

def rotation_vector_to_quaternion(rvec):
    """
    Converte un vettore di rotazione (Rodrigues) in un quaternion [qx, qy, qz, qw].
    """
    # rvec: array di forma (3,) o (3,1)
    R, _ = cv2.Rodrigues(rvec)
    # Calcolo trace:
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2  # S = 4*qw
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        # Caso t <= 0: prendere il maggior fra R[0,0],R[1,1],R[2,2]
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2  # S = 4*qx
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2  # S=4*qy
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # S=4*qz
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    # Normalizziamo:
    return q / np.linalg.norm(q)

def quaternion_slerp(q1, q2, fraction):
    """
    SLERP (Spherical Linear Interpolation) tra due quaternioni unitari q1 e q2,
    con fraction ∈ [0,1]. Ritorna un quaternion unitario.
    """
    # Assicurarsi che siano unitari:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    # Se negativo, cambiarne il segno per prendere il “cammino breve”
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Se quasi identici, facciamo interpolation lineare e normalizziamo
        result = q1 + fraction * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)        # angolo tra q1 e q2
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * fraction      # angolo a “fraction”    
    sin_theta = np.sin(theta)
    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return (s1 * q1) + (s2 * q2)

def invert_transform(rvec, tvec):
    """
    Inverte la trasformazione (rvec, tvec) → (rvec_inv, tvec_inv),
    cioè da Camera→Board a Board→Camera.
    """
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv.dot(tvec.reshape(3,))
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv.flatten(), t_inv.flatten()

def apply_pose_correction(rvec, tvec, mode="rot_z_-90"):
    
    # assumo che rvec sia in formato rodrigues e creo matrice di rotazione
    R, _ = cv2.Rodrigues(rvec)
    # estendere matrice di rotazione 3x3 a 4x4 (ultima colonna contenente i tvec)

    # 

    if mode == "rot_z_90":
        theta = np.radians(90)
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        R_corr = R @ Rz
        t_corr = Rz.T @ tvec.reshape(3,)

    elif mode == "rot_z_-90":
        theta = np.radians(-90)
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        R_corr = R @ Rz
        t_corr = Rz.T @ tvec.reshape(3,)

    else:
        R_corr = R
        t_corr = tvec.reshape(3,)

    rvec_corr, _ = cv2.Rodrigues(R_corr)
    return rvec_corr, t_corr