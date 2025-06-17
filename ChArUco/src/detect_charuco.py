import cv2
import numpy as np

# We always use the same dictionary when printing the boards
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

def create_charuco_boards(board_size, board_physical_size, marker_length_ratio):
    """
    Create two non-overlapping CharucoBoards, with different IDs (ids1 and ids2)
    similar to how you generated them:
    - board_size: tuple (N, N), e.g. (3,3) or (5,5)
    - board_physical_size: side in meters of the entire printed board (here 0.075 m)
    - marker_length_ratio: ratio marker_length / square_length
    Returns (board1, board2, square_length, marker_length).
    """
    squares_x, squares_y = board_size
    # ---- Dynamic square_length calculation: whole board makes board_physical_size m
    square_length = board_physical_size / squares_x
    # marker_length is a fraction of square_length
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
    Attempts to detect a single CharucoBoard in img_gray.
    Returns (rvec, tvec) if successful, otherwise None.
    """
    
    # Crea un oggetto DetectorParameters per la rilevazione dei marker ArUco
    par = cv2.aruco.DetectorParameters()

    # === Sub-pixel refinement settings ===

    # Metodo per il raffinamento dei corner:
    # Attiva il raffinamento sub-pixel con cornerSubPix di OpenCV.
    # Migliora la precisione della posizione dei corner dei marker.
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Default: CORNER_REFINE_NONE

    # Numero massimo di iterazioni del processo di raffinamento:
    # Più iterazioni permettono un affinamento più accurato, ma aumentano il tempo di elaborazione.
    par.cornerRefinementMaxIterations = 60  # Default: 30

    # Accuratezza minima per terminare le iterazioni:
    # L'algoritmo si ferma se il miglioramento è inferiore a questa soglia.
    # Valori più bassi danno una precisione maggiore, ma aumentano il tempo di calcolo.
    par.cornerRefinementMinAccuracy = 0.05  # Default: 0.1

    # Dimensione della finestra di ricerca (in pixel) usata per cercare il massimo sub-pixel:
    # La finestra reale sarà 2*WinSize + 1 (es: 7 → 15x15 pixel).
    # Finestra più grande = più accuratezza, ma rischio di confusione se l'immagine è rumorosa.
    par.cornerRefinementWinSize = 7  # Default: 5

    # 1. detect ArUco markers:
    corners, ids, _ = cv2.aruco.detectMarkers(
        img_gray, ARUCO_DICT, parameters= par
    )
    if ids is None or len(ids) == 0:
        return None
    # 2. find Charuco corners:
    
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=img_gray,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        # min 4 corners to solvePnP to be robust
        return None

    #3. estimate the pose of the CharucoBoard 
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
    Given a BGR image, try to detect board1 first then board2.
    Returns:
    results = [
    (marker_id, rvec, tvec), # e.g. ("C1", rvec1, tvec1)
    (marker_id, rvec, tvec) # e.g. ("C2", rvec2, tvec2)
    ]
    If either is NOT found, it does NOT appear in the list.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    out1 = detect_single_charuco(img_gray, board1, camera_matrix, dist_coeffs)
    if out1 is not None:
        # Let's assign “C1” to the first board:
        rvec1, tvec1 = out1
        results.append(("C1", rvec1, tvec1))
    out2 = detect_single_charuco(img_gray, board2, camera_matrix, dist_coeffs)
    if out2 is not None:
        # Let's assign “C2” to the second board:
        rvec2, tvec2 = out2
        results.append(("C2", rvec2, tvec2))
    return results
