# src/main.py
import os
import cv2
import glob
import time
import argparse
import numpy as np
import csv

from utils import load_camera_calibration, rotation_vector_to_quaternion, apply_pose_correction
from detect_charuco import create_charuco_boards, detect_two_charuco
from plate_pose import relative_pose

# DISTANZA REALE TRA I MARKER: ipotenusa di 110 mm su X e Y (≈ 155.6 mm)
EXPECTED_DISTANCE_M = 0.15556349 #np.sqrt(0.11**2 + 0.11**2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rileva due CharucoBoard in ogni immagine e stima la posa del plate."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Cartella contenente le immagini da processare"
    )
    parser.add_argument(
        "--board_size", required=True, type=int, choices=[3,5],
        help="Dimensione della CharucoBoard: 3 o 5 (3×3 o 5×5)"
    )
    parser.add_argument(
        "--calib_file", required=True,
        help="File .yaml con la calibrazione camera (OpenCV)"
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="File CSV dove salvare tutte le pose"
    )
    # Se vuoi cambiare il rapporto marker_length/square_length default:
    parser.add_argument(
        "--marker_length_ratio", type=float, default=0.8,
        help="rapporto (marker_length / square_length). Modificare se necessario."
    )
    parser.add_argument(
    "--debug", action="store_true",
    help="Mostra le immagini con gli assi dei marker rilevati (usa ESC per uscire)"
)

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Carico calibrazione camera
    camera_matrix, dist_coeffs = load_camera_calibration(args.calib_file)

    # 2. Creo i due CharucoBoard
    board_size = (args.board_size, args.board_size)
    board_physical_size = 0.075  # in metri (75 mm fisici)
    marker_length_ratio = args.marker_length_ratio
    board1, board2, square_length, marker_length = create_charuco_boards(
        board_size, board_physical_size, marker_length_ratio
    )

    # 3. Preparo il CSV di output (header + modalità append)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_file = open(args.output_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "image_name",
        "tx_rel", "ty_rel", "tz_rel",
        "distance_m", "error_mm",
        "qx_rel", "qy_rel", "qz_rel", "qw_rel",
        "elapsed_time_s"
    ])

    # 4. Elenco immagini
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    total = len(img_paths)
    print(f"[INFO] Trovate {total} immagini in {args.input_dir}")

    for idx, img_path in enumerate(img_paths, start=1):
        start_t = time.time()
        img_name = os.path.basename(img_path)

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise IOError(f"Impossibile leggere {img_path}")
        except Exception as e:
            print(f"[WARNING] Immagine {img_name}: errore lettura → salto. ({e})")
            continue

        # 4.1. Rilevo marker
        detected = detect_two_charuco(img_bgr, board1, board2, camera_matrix, dist_coeffs)
        if len(detected) != 2:
            print(f"[WARNING] {img_name}: rilevati {len(detected)} marker (ne servono 2) → salto.")
            continue

        # 4.2. Recupero pose
        pose_dict = {marker_id: (rvec, tvec) for marker_id, rvec, tvec in detected}
        if not ("C1" in pose_dict and "C2" in pose_dict):
            print(f"[WARNING] {img_name}: mancano marker C1 o C2 → salto.")
            continue

        rvec1, tvec1 = pose_dict["C1"]
        rvec2, tvec2 = pose_dict["C2"]

        # Applica correzione al secondo marker
        #rvec2, tvec2 = apply_pose_correction(rvec2, tvec2, mode="rot_z_-90")
        
        if args.debug:
            debug_img = img_bgr.copy()
            axis_length = 0.03  # 3 cm
            cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec1, tvec1, axis_length)
            cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec2, tvec2, axis_length)

            # Mostra angolo tra gli assi Z
            R1, _ = cv2.Rodrigues(rvec1)
            R2, _ = cv2.Rodrigues(rvec2)
            z1 = R1[:, 2]
            z2 = R2[:, 2]
            angle_z = np.degrees(np.arccos(np.clip(np.dot(z1, z2), -1.0, 1.0)))
            print(f"[DEBUG] Angolo tra Z1 e Z2 = {angle_z:.1f}°")

            # Ridimensionamento e finestra interattiva
            scale_factor = 0.5
            debug_resized = cv2.resize(debug_img, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)
            cv2.imshow("DEBUG", debug_resized)
            cv2.resizeWindow("DEBUG", 960, 720)
            cv2.moveWindow("DEBUG", 100, 100)

            key = cv2.waitKey(0)
            if key == 27:  # ESC per uscire
                cv2.destroyAllWindows()
                exit()

        # 4.3. Calcolo trasformazione relativa (C1 → C2)
        rvec_rel, tvec_rel = relative_pose(rvec1, tvec1, rvec2, tvec2)
        distance = np.linalg.norm(tvec_rel[:2])
        error_mm = (distance - EXPECTED_DISTANCE_M) * 1000.0
        q_rel = rotation_vector_to_quaternion(rvec_rel)
        elapsed = time.time() - start_t

        # 4.4. Scrivo su CSV
        csv_writer.writerow([
            img_name,
            tvec_rel[0], tvec_rel[1], tvec_rel[2],
            distance, error_mm,
            q_rel[0], q_rel[1], q_rel[2], q_rel[3],
            f"{elapsed:.4f}"
        ])

        print(f"[{idx}/{total}] {img_name} → Δ= {distance:.4f} m (err={error_mm:+.1f} mm)")

    csv_file.close()
    print(f"[DONE] Output salvato in: {args.output_csv}")

if __name__ == "__main__":
    main()
