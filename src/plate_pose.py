# src/plate_pose.py
import numpy as np
import cv2
from utils import rotation_vector_to_quaternion, quaternion_slerp

def board_center_from_rvec_tvec(rvec, tvec, board_size, square_length):
    """
    Dato il rvec/tvec del CharucoBoard (che OpenCV definisce con origine
    nell’angolo inferiore-sinistro del board, coordinate 0,0,0),
    calcola rvec_center/tvec_center in cui l’origine è posta al centro del
    board (board_size = (squares_x, squares_y), square_length in metri).

    Restituisce (rvec_center, tvec_center).
    """
    squares_x, squares_y = board_size
    width = squares_x * square_length
    height = squares_y * square_length
    # Coordinate del centro, nel sistema di riferimento del board:
    offset_board = np.array([width/2.0, height/2.0, 0.0], dtype=np.float64)

    # Prima otteniamo la rotazione 3×3 e poi applichiamo l’offset:
    R_board, _ = cv2.Rodrigues(rvec)
    t_board = tvec.reshape(3,)
    # t_center = R_board * offset_board + t_board
    t_center = R_board.dot(offset_board) + t_board
    # la rotazione resta la stessa:
    r_center, _ = cv2.Rodrigues(R_board)
    return r_center.flatten(), t_center.flatten()

def fuse_two_poses(rvec1, tvec1, rvec2, tvec2):
    """
    Riceve (rvec1, tvec1) e (rvec2, tvec2), già riferiti al centro del board.
    Calcola la fusione come:
      - traslazione = media aritmetica di t1, t2
      - rotazione = SLERP(q1, q2, 0.5)
    Restituisce (rvec_fused, tvec_fused).
    """
    # 1. traslazione media:
    t1 = tvec1.reshape(3,)
    t2 = tvec2.reshape(3,)
    t_fused = 0.5 * (t1 + t2)

    # 2. rotazioni → quaternioni:
    from utils import rotation_vector_to_quaternion
    q1 = rotation_vector_to_quaternion(rvec1)
    q2 = rotation_vector_to_quaternion(rvec2)
    q_fused = quaternion_slerp(q1, q2, 0.5)  # metà strada

    # 3. riconvertiamo q_fused in matrice 3×3, poi in rvec:
    qw = q_fused[3]
    qx, qy, qz = q_fused[0], q_fused[1], q_fused[2]
    # Costruiamo manualmente la matrice di rotazione:
    Rf = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),   1 - 2*(qx*qx + qz*qz),       2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),   1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    rvec_fused, _ = cv2.Rodrigues(Rf)
    return rvec_fused.flatten(), t_fused

def relative_pose(rvec_from, tvec_from, rvec_to, tvec_to):
    """
    Calcola la trasformazione relativa da marker1 (FROM) a marker2 (TO).
    Ritorna: (rvec_rel, tvec_rel)
    """
    R_from, _ = cv2.Rodrigues(rvec_from)
    R_to, _ = cv2.Rodrigues(rvec_to)
    R_from_inv = R_from.T
    t_from_inv = -R_from_inv @ tvec_from.reshape(3,)
    R_rel = R_from_inv @ R_to
    t_rel = R_from_inv @ (tvec_to.reshape(3,) - tvec_from.reshape(3,))
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    return rvec_rel.flatten(), t_rel.flatten()
