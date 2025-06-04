# src/detect_charuco.py
import cv2
import numpy as np

# Usiamo sempre lo stesso dizionario:
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

def create_charuco_boards(board_size, board_physical_size, marker_length_ratio):
    """
    Crea due CharucoBoard non sovrapposte, con ID differenti (ids1 e ids2)
    in modo analogo a come hai generato tu:
      - board_size: tuple (N, N), es. (3,3) oppure (5,5)
      - board_physical_size: lato in metri dell’intera board stampata (qui 0.075 m)
      - marker_length_ratio: rapporto marker_length / square_length
    Ritorna (board1, board2, square_length, marker_length).
    """
    squares_x, squares_y = board_size
    # ---- Calcolo square_length dinamico: tutta la tavola fa board_physical_size m
    square_length = board_physical_size / squares_x
    # marker_length è una frazione di square_length
    marker_length = square_length * marker_length_ratio

    # numero di marker ArUco (floor(N*N/2))
    n_markers = (squares_x * squares_y) // 2
    ids1 = np.arange(0, n_markers, dtype=int)
    ids2 = np.arange(n_markers, 2 * n_markers, dtype=int)

    board1 = cv2.aruco.CharucoBoard(
        board_size,      # (squares_x, squares_y)
        square_length,   # lato quadrato (metri)
        marker_length,   # lato marker (metri)
        ARUCO_DICT,      # il dizionario 5×5_100
        ids1             # ID usati in board1
    )
    board2 = cv2.aruco.CharucoBoard(
        board_size,
        square_length,
        marker_length,
        ARUCO_DICT,
        ids2
    )
    return board1, board2, square_length, marker_length

def detect_single_charuco(img_gray, board, camera_matrix, dist_coeffs):
    """
    Prova a rilevare un singolo CharucoBoard in img_gray.
    Ritorna (rvec, tvec) se riuscito, altrimenti None.
    """
    # 1. rileva marker ArUco:
    corners, ids, _ = cv2.aruco.detectMarkers(
        img_gray, ARUCO_DICT, 

    )
    if ids is None or len(ids) == 0:
        return None
    # 2. trova i Charuco corners:
    
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=img_gray,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        # minimo 4 corner per solvePnP per essere robusti
        return None

    # 3. stima la posa del CharucoBoard  
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        rvec=None,
        tvec=None
    )
    if not success:
        return None
    return rvec.flatten(), tvec.flatten()

def detect_two_charuco(img_bgr, board1, board2, camera_matrix, dist_coeffs):
    """
    Data un’immagine BGR, prova a rilevare prima board1 poi board2.
    Restituisce:
      results = [
         (marker_id, rvec, tvec),   # es. ("C1", rvec1, tvec1)
         (marker_id, rvec, tvec)    # es. ("C2", rvec2, tvec2)
      ]
    Se uno dei due NON viene trovato, NON appare nella lista.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    out1 = detect_single_charuco(img_gray, board1, camera_matrix, dist_coeffs)
    if out1 is not None:
        # Assegniamo “C1” al primo board:
        rvec1, tvec1 = out1
        results.append(("C1", rvec1, tvec1))
    out2 = detect_single_charuco(img_gray, board2, camera_matrix, dist_coeffs)
    if out2 is not None:
        # Assegniamo “C2” al secondo board:
        rvec2, tvec2 = out2
        results.append(("C2", rvec2, tvec2))
    return results
