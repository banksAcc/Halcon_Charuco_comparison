import cv2
import numpy as np
import cv2.aruco as aruco
import os
import json

def calibrate_from_images(image_folder, output_json_path):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    board = aruco.CharucoBoard((12, 9), 30, 22, aruco_dict)

    all_corners = []
    all_ids = []
    image_size = None

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    if len(image_files) == 0:
        print("❌ Nessuna immagine trovata nella cartella.")
        return

    for filename in image_files:
        path = os.path.join(image_folder, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Immagine non valida: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

        if ids is not None and len(ids) > 0:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

            if retval is not None and retval > 20:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"✅ Rilevata board valida in: {filename}")
                if image_size is None:
                    image_size = gray.shape[::-1]
            else:
                print(f"❌ Board non sufficientemente visibile in: {filename}")
        else:
            print(f"❌ Nessun marker ArUco trovato in: {filename}")

    if len(all_corners) < 5:
        print("❌ Pochi frame validi per calibrazione.")
        return

    print("⚙️ Calibrazione in corso...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    np.savez("data/calib_data.npz",
                 cameraMatrix=camera_matrix,
                 distCoeffs=dist_coeffs,
                 rvecs=rvecs,
                 tvecs=tvecs,
                 rms=rms)
    
    # Estrazione dei parametri nel formato desiderato
    k = dist_coeffs.flatten()
    calib_data = {
        "IntrinsicCalibration": {
            "OpenCV": {
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "k1": float(k[0]),
                "k2": float(k[1]),
                "p1": float(k[2]),
                "p2": float(k[3]),
                "k3": float(k[4]),
                "k4": float(k[5]),
                "k5": float(k[6]),
                "k6": float(k[7]),
                "rms": float(rms)
            }
        }
    }

    with open(output_json_path, 'w') as f:
        json.dump(calib_data, f, indent=4)

    print(f"✅ Calibrazione completata. Dati salvati in {output_json_path}")

# ESEMPIO USO:
calibrate_from_images("../../calibration/calib_charuco", "calib_output.json")
