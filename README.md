## Vision-Based Memory Game System

A computer vision pipeline that assists the classic Memory card game by detecting cards from images, determining whether each card is face up or face down, matching pairs using image features, and managing game flow and scoring. The project contains a reusable function module and a demonstration notebook that runs the full pipeline on a sequence of images representing the game.

### Key Features
- **Data preparation**: Global scaling based on maximum pixel count; contrast enhancement via CLAHE.
- **Card detection**: Canny edges + morphology + contour filtering with area/aspect heuristics.
- **Back template creation**: Averages detected card regions to build a card-back template.
- **State classification**: Classifies face up/down via template correlation, color variance, and color-distance checks.
- **Card matching**: SIFT features + BFMatcher with Lowe's ratio test; adaptive match threshold based on keypoint counts.
- **Game state management**: Tracks turns, matched pairs, current player, and scores; produces readable instructions.
- **Visualization**: Draws labeled bounding boxes and highlights attempted matches.
- **Evaluation utilities**: Detection count accuracy, classification accuracy, and matching accuracy helpers.

## Repository Structure
- `functions.py`: Core vision and game-logic utilities (detection, classification, matching, visualization, metrics).
- `Project_Memory.ipynb`: End-to-end demo and evaluation workflow using images of game turns.
- `data/` (not tracked): Expected folder for input images (see Dataset section).

## Requirements
- Python 3.9+ (recommended)
- Packages:
  - `opencv-contrib-python` (SIFT support)
  - `numpy`
  - `matplotlib`

Install via pip:
```bash
pip install opencv-contrib-python numpy matplotlib
```

## Dataset
Place images under a folder such as `data/` (or adjust the path in the notebook):
- `initial.png`: Board before any turn
- `1.png` ... `14.png`: Turn images (edit the range as needed)
- `end.png`: Board at the end

In the notebook, the base directory is controlled by `IMG_DIR`, e.g.:
```python
IMG_DIR = "/path/to/your/data/"
```

## How It Works (Pipeline Overview)
1. Load and preprocess images with global scaling and CLAHE enhancement.
2. Detect card bounding boxes via edges → morphology → contours → area/aspect filtering.
3. Build a card-back template from detected regions in the initial image.
4. Classify each card as face up/down using template correlation and color statistics.
5. When two cards are face up, compare them using SIFT feature matching to decide if they form a pair.
6. Update the game state (whose turn, matched pairs, scores) and visualize results.

## Quick Start
Open `Project_Memory.ipynb` and run cells top-to-bottom. Adjust `IMG_DIR` to your dataset path. The notebook will:
- Initialize the game
- Detect cards and build the back template
- Iterate through turn images, classify states, attempt matches
- Visualize each step and print game instructions/scores
- Run small evaluation examples

## Evaluation Helpers
The notebook defines `evaluate_turn(...)` to compute metrics for a given image:
- **Detection Count Accuracy**: Compares detected count with ground-truth count.
- **Classification Accuracy**: Compares predicted face up/down vs ground truth.
- **Matching Accuracy**: Compares predicted match vs ground-truth match for a pair.

## Notes and Tips
- SIFT requires `opencv-contrib-python`. If SIFT fails, check your OpenCV installation.
- The contour-area heuristics assume a reasonable number of cards in view; adjust thresholds if your setup differs.
- Lighting and reflections can affect classification; the combination of template matching and color statistics is designed to be robust, but you may need to tweak thresholds for your data.

## Acknowledgements
Developed for an image analysis project demonstrating classical computer vision techniques applied to a Memory game scenario.
