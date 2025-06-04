# src/main.py
import os
import cv2
import glob
import time
import argparse
import numpy as np
import csv

from utils import load_camera_calibration, rotation_vector_to_quaternion
from detect_charuco import create_charuco_boards, detect_two_charuco
from plate_pose import board_center_from_rvec_tvec, fuse_two_poses

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
        "image_name", "marker_id",
        "tx", "ty", "tz",
        "qx", "qy", "qz", "qw",
        "elapsed_time_s"
    ])

    # 4. Itero su tutte le immagini nella cartella (png/jpg/png)
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
            print(f"[WARNING] Immagine {img_name}: errore di lettura → salto. ({e})")
            continue

        # 4.1. rilevo i due Charuco (C1, C2)
        detected = detect_two_charuco(img_bgr, board1, board2, camera_matrix, dist_coeffs)
        if len(detected) != 2:
            print(f"[WARNING] {img_name}: rilevati {len(detected)} marker (ne servono 2) → salto fusione.")
            # Se vuoi scrivere comunque le righe per i marker singoli:
            for (marker_id, rvec, tvec) in detected:
                q = rotation_vector_to_quaternion(rvec)
                elapsed = time.time() - start_t
                csv_writer.writerow([
                    img_name, marker_id,
                    tvec[0], tvec[1], tvec[2],
                    q[0], q[1], q[2], q[3],
                    f"{elapsed:.4f}"
                ])
            continue

        # 4.2. per ogni marker calcolo (rvec_center, tvec_center)
        centers = {}
        for (marker_id, rvec, tvec) in detected:
            r_center, t_center = board_center_from_rvec_tvec(
                rvec, tvec, board_size, square_length
            )
            centers[marker_id] = (r_center, t_center)
            # Scrivo la riga “marker singolo”:
            q = rotation_vector_to_quaternion(r_center)
            elapsed = time.time() - start_t
            csv_writer.writerow([
                img_name, marker_id,
                t_center[0], t_center[1], t_center[2],
                q[0], q[1], q[2], q[3],
                f"{elapsed:.4f}"
            ])

        # 4.3. ora i due marker ⇒ fusione
        rvec1, tvec1 = centers["C1"]
        rvec2, tvec2 = centers["C2"]
        r_fused, t_fused = fuse_two_poses(rvec1, tvec1, rvec2, tvec2)
        qf = rotation_vector_to_quaternion(r_fused)
        elapsed = time.time() - start_t
        csv_writer.writerow([
            img_name, "Fusion",
            t_fused[0], t_fused[1], t_fused[2],
            qf[0], qf[1], qf[2], qf[3],
            f"{elapsed:.4f}"
        ])

        # 4.4. stato avanzamento
        print(f"[{idx}/{total}] Processata {img_name} → Fusion ({t_fused[0]:.3f}, {t_fused[1]:.3f}, {t_fused[2]:.3f}); dt={elapsed:.3f}s")

    csv_file.close()
    print(f"[DONE] Tutte le pose salvate in {args.output_csv}")

if __name__ == "__main__":
    main()
