import os
import cv2
import glob
import time
import numpy as np
import csv

from utils import load_camera_calibration,pose_to_matrix, offset_pose_to_center, rotation_matrix_to_quaternion, matrix_to_pose, parse_args_from_json
from detect_charuco import create_charuco_boards, detect_two_charuco

# DISTANZA REALE TRA I MARKER: ipotenusa di 110 mm su X e Y (≈ 155.6 mm)
EXPECTED_DISTANCE_M = 0.15556349 #np.sqrt(0.11**2 + 0.11**2)

def main():
    args = parse_args_from_json()

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
        "tx_rel_mm", "ty_rel_mm", "tz_rel_mm",
        "distance_mm", "error_mm",
        "qx_rel_mm", "qy_rel_mm", "qz_rel_mm", "qw_rel_mm",
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

        # 1. Conversione in matrici 4x4
        T1 = pose_to_matrix(rvec1, tvec1)
        T2 = pose_to_matrix(rvec2, tvec2)

        # 2. Applica offset per portarsi al centro della board
        offset = np.array([0.0375, 0.0375, 0.0])
        T1_center = offset_pose_to_center(T1, offset)
        T2_center = offset_pose_to_center(T2, offset)

        # 3. Trasformazione relativa
        T_rel = np.linalg.inv(T1_center) @ T2_center
        t_rel = T_rel[:3, 3]
        R_rel = T_rel[:3, :3]
        distance = np.linalg.norm(t_rel)

        # 4. Quaternione
        q_rel = rotation_matrix_to_quaternion(R_rel)

        if args.debug:
            debug_img = img_bgr.copy()
            axis_length = 0.035  #  3,5 cm

            # Converti la posa centrata in rvec/tvec
            rvec1_center, tvec1_center = matrix_to_pose(T1_center)
            rvec2_center, tvec2_center = matrix_to_pose(T2_center)

            # Disegna gli assi dal centro reale della board
            cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec1_center, tvec1_center, axis_length)
            cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec2_center, tvec2_center, axis_length)

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

        elapsed = time.time() - start_t

        # Converti traslazione e distanza in mm
        t_rel_mm = (t_rel * 1000.0).tolist()        # tx, ty, tz in mm
        distance_mm = distance * 1000.0             # distanza in mm
        error_mm = distance_mm - (EXPECTED_DISTANCE_M * 1000.0)

        # 5. Salvataggio CSV (tutto in mm)
        csv_writer.writerow([
            img_name,
            *t_rel_mm,
            distance_mm,
            error_mm,
            *q_rel.tolist(),
            f"{elapsed:.4f}"
        ])

        print(f"[{idx}/{total}] {img_name} → Δ= {distance:.4f} m (err={error_mm:+.1f} mm)")

    csv_file.close()
    print(f"[DONE] Output salvato in: {args.output_csv}")

if __name__ == "__main__":
    main()
