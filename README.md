# 🔧 Charuco Dual Marker Pose Estimation

This project estimates the **relative 6DoF pose** between two [ChArUco boards](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html) placed diagonally on a square plate. The system uses OpenCV’s ArUco module and a calibrated Basler camera to compute pose and orientation for each marker, calculates their **center-to-center distance**, and exports the result in a CSV.

The pipeline is designed for comparison with Halcon-based systems in industrial computer vision applications.

---

## 📌 Project Overview

- Two ChArUco boards (3×3 or 5×5) are placed on a 75 mm × 75 mm 3D-printed plate.
- The markers are placed diagonally, 4 cm from the edges.
- Pose is estimated using OpenCV and corrected to point to the **center** of each board.
- The **relative transformation** is extracted and exported (translation in mm, orientation as quaternions).

---

## 📁 Directory Structure

```

├── src/
│   ├── main.py                # Main script
│   ├── detect_charuco.py      # ChArUco detection logic
│   ├── utils.py               # Pose/matrix utilities
│   └── settings.json          # Run-time parameters
│ 
├── calibration/
│   └── camera_calib.yaml      # Camera intrinsics
│
├── data/
│
├── output/
│   └── results.csv            # Output results
│ 
├── requirements.txt
│
└── README.md

```

---

## 🧪 Example Result (CSV Columns)

- `tx_rel_mm`, `ty_rel_mm`, `tz_rel_mm`: translation in mm
- `distance_mm`: total center-to-center distance
- `error_mm`: deviation from expected physical value (≈ 155.6 mm)
- `qx_rel`, `qy_rel`, `qz_rel`, `qw_rel`: quaternion orientation
- `elapsed_time_s`: processing time

---

## ⚙️ How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
```

> Ensure you use `opencv-contrib-python` ≥ 4.11.0

### 2. Prepare the configuration

Edit `settings.json`:

```json
{
  "input_dir": "../data/charuco5x5",
  "board_size": 5,
  "calib_file": "../../calibration/camera_calib_opencv.yaml",
  "output_csv": "output/results.csv",
  "marker_length_ratio": 0.752,
  "debug": false
}
```

### 3. Run the project

```bash
python src/main.py
```

> Images will be processed from the directory set in `input_dir`.

---

## 🧰 Debug Mode

To visually inspect the detected poses:

```json
"debug": true
```

Then run again:

```bash
python src/main.py
```

Press `ESC` to continue through images.

---

## 📦 Dataset (via Hugging Face)

We provide a test dataset with real camera acquisitions:

📁 **[🧬 View and download on Hugging Face →](https://huggingface.co/datasets/banksAcc/Halcon_Charuco_comparison)**  
(Size: ~240MB, TIFF format, Basler camera)

After downloading, place the dataset under:

```
../data/charuco5x5/
```

or

```
../data/charuco3x4/
```

Or edit `settings.json` to match your folder.

---

## 📐 Board Dimensions

- Square size: `25 mm`
- Marker size: `19 mm`
- `marker_length_ratio`: `0.76`
- Physical distance between marker centers: `~155.56349 mm`

---

### 📏 Gauge Capability Indexes (Cg & Cgk)

To evaluate the **metrological performance** of the pose estimation system, two standard capability indices can be used:

- **Cg (Gauge Capability)**  
  Represents the **intrinsic capability** of the measurement system (in this case, the camera + pose estimation algorithm).  
  It reflects **repeatability and precision**, but **ignores any bias** from the real value.  
  A high Cg means the system consistently produces tightly grouped results.

- **Cgk (Corrected Gauge Capability)**  
  Like Cg, but also includes the **systematic error (bias)** between the measured and real value.  
  It indicates **how reliable** the measurement is with respect to the actual reference (e.g., 155.6 mm center-to-center distance).  
  A high Cgk confirms both precision and **accuracy**.

These indices are commonly used in **gauge R&R (repeatability and reproducibility)** studies and can support quality control or certification processes in industrial settings.

---
## 📄 License

MIT License – Free for commercial and academic use.

---

## 🙋‍♂️ Author

Developed by Angelo Milella, for comparative analysis between **OpenCV** and **Halcon**-based pose estimation systems. Projetc From Comau
