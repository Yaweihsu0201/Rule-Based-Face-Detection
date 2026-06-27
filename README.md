# Face Detection with Skin Segmentation and Facial Landmark Triplet Scoring

This project implements a face detection pipeline inspired by the method proposed in:

> Ke-Jie Liao 廖科傑, "運用輪廓色彩和五官的人臉偵測技術 Face Detection by Outline, Color, and Facial Features", 2010.

This project implements a face detection pipeline based on skin segmentation, ellipse matching, facial feature maps, geometric filtering, and machine learning-based triplet scoring.

Compared with the original method, this implementation includes several modifications, including:

- Replace color-based skin filtering with Deep Learning method
- Add several more eye-mouth features into consideration
- Select the most likely facial landmark triplet using geometric constraints and a trained Random Forest classifier.

[View Project Report](docs/ADSP_Final.pdf)
## Overview

The goal of this project is to detect faces in an image by identifying the best combination of:

- left eye
- right eye
- mouth

within candidate face ellipses.

The final output visualizes the detected facial structure by drawing:

- the fitted face ellipse
- detected eye and mouth points
- lines connecting the eyes and mouth

## Project Structure

```text
Rule-Based-Face-Detection/
│
├── main.py
├── detector.py
│
├── utilities/
│   ├── predict_skin.py
│   ├── ellipse_matching.py
│   ├── eyemap.py
│   ├── mouthmap.py
│   ├── triplet_finder.py
│   ├── triplet_scorer.py
│   ├── triplet_geometry.py
│   └── visualization.py
│
├── checkpoints/
│   ├── unet_skin_best.pth
│   └── triplet_scorer_rf.pkl
│
├── TestImagesForPrograms/
├── FaceDetectionResults/
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Yaweihsu0201/Rule-Based-Face-Detection.git
cd Face-Detection
```

Create a virtual environment:

```bash
py -3.10 -m venv .venv
```

Activate the virtual environment:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run face detection on a single image:

```bash
python main.py --input TestImagesForPrograms/23.jpg --output FaceDetectionResults/23.png
```

Arguments:

```text
--input     Path to the input image
--output    Path to save the detection result
```

Example:

```bash
python main.py --input TestImagesForPrograms/001.jpg --output FaceDetectionResults/001.png
```

## Results example

| Original Image | Detection Result |
|---------------|----------------|
| ![](docs/19.jpg) | ![](docs/19.png) |

