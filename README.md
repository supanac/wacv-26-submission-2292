# wacv-26-submission-2292

# Automatic Video Analysis and Annotation Framework

A framework for automatic video analysis and annotation, specifically designed for the study of co-speech gesture in linguistics and beyond. This framework integrates and automates the usage of several existing components, including PySceneDetect for scene detection, OpenPose for body pose estimation, and an optimized version of TalkNet-ASD for active speaker detection.

On top of the post-processed machine learning output, we developed a rule-based automatic annotation system in close dialogue between computer vision scholars and linguists. This system derives hand positions relative to the body, speed and direction of the hands, and generates standardised outputs compatible with ELAN and CQPweb (standard software in linguistic research), allowing both for detailed gesture analysis and for corpus-based research.

## Features

- **Automatic annotation generation** from pose keypoints.
- **ELAN file creation** with customizable tiers and annotations.
- **Timeseries extraction** for gesture analysis.
- **Rule-based labeling** for spatial positions, movement directions, and speeds.
- **Integration with linguistic research tools** (ELAN, CQPweb).
- **Configurable processing** via YAML configuration files.

## Installation

### Dependencies

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

### Additional Requirements

- **OpenPose** for keypoint extraction (preprocessing step).
- **PySceneDetect** for scene detection (preprocessing step).
- [**TalkNet-ASD**](TalkNet-ASD/) for active speaker detection (preprocessing step).

## Quick Start

### 1. Prepare Your Data

Ensure you have the following input files:
- Keypoint data in NPZ format (`video_id.npz`).
- Scene boundaries (CSV format or included in NPZ file).
- Modify configuration files in `configs/` directory

### 2. Update the Bash Script

Edit the paths in `run_analysis.sh`:

```bash
#!/bin/bash

VIDEO_ID="$1"
VIDEO_FPS="$2"

# Update these paths for your system
TIMESERIES_HOME=/path/to/keypoints.npz
PATH_TO_SCENES=/to/path/scenes.csv
SAVE_TO=/where/to/save/results/

python pseudolabels.py \
    --timeseries-home "$TIMESERIES_HOME" \
    --path-to-scenes "$PATH_TO_SCENES" \
    --video-id "$VIDEO_ID" \
    --video-fps "$VIDEO_FPS"

python create_tiers.py \
    --timeseries-home "$TIMESERIES_HOME" \
    --path-to-scenes "$PATH_TO_SCENES" \
    --video-id "$VIDEO_ID" \
    --video-fps "$VIDEO_FPS" \
    --save-to "$SAVE_TO"
```

### 3. Run the Analysis

```bash
chmod +x run.sh
./run.sh your_video_id 30.0
```

## Usage

### Two-Step Process

The framework operates in two main steps:

#### Step 1: Generate Annotations (`pseudolabels.py`)
Creates rule-based annotations from keypoint data and saves them to an NPZ file.

```bash
python pseudolabels.py \
    --timeseries-home /path/to/keypoints/ \
    --path-to-scenes /path/to/scenes.csv \
    --video-id your_video_id \
    --video-fps 30.0 \
    --annotation-fn custom_annotation_name  # optional
```

#### Step 2: Create ELAN Files (`create_tiers.py`)
Generates ELAN-compatible files with annotations and timeseries data.

```bash
python create_tiers.py \
    --timeseries-home /path/to/keypoints/ \
    --path-to-scenes /path/to/scenes.csv \
    --video-id your_video_id \
    --video-fps 30.0 \
    --save-to /path/to/output/ \
    --eaf-file existing_file.eaf \        # optional, creates new if not provided
    --annotation-fn custom_annotation_name # optional
```

### Independent Usage

- **If you already have annotation files**: Run only `create_tiers.py`
- **If you need fresh annotations**: Run both scripts in sequence

## Configuration

### Pseudolabels Configuration (`configs/pseudolabels_config.yaml`)

Defines which annotation tiers to generate:

```yaml
wrists:
  right: ["annotation_x", "annotation_y", "zones"]
  left: ["annotation_x", "annotation_y", "zones"]
fingers:
  right:
    thumb: ["annotation_x", "annotation_y"]
    index: ["annotation_x", "annotation_y"]
    # ... other fingers
  left:
    # ... similar structure
eyelid:
  right: ["annotation"]
  left: ["annotation"]
```

### ELAN Configuration (`configs/elan_config.yaml`)

Specifies tiers and timeseries for ELAN output:

```yaml
tiers:
  - "right_wrist_annotation_x"
  - "left_wrist_annotation_y"
  - "right_index_finger_annotation_x"
  # ... other tiers

timeseries:
  - "Right Wrist Lateral Position"
  - "Right Wrist Vertical Position"
  - "Left Wrist Velocity"
  # ... other timeseries
```

## Input Data Formats

### Keypoint Data (NPZ file)
Expected structure in `video_id.npz`:
- `pose_keypoints_2d_norm`: Normalized 2D pose keypoints.
- `hand_left_keypoints_2d_norm`: Normalized left hand keypoints.
- `hand_right_keypoints_2d_norm`: Normalized right hand keypoints.
- `face_keypoints_2d_norm`: Normalized face keypoints.
- `confidence_scores`: Keypoint confidence values.
- `boundaries`: Scene boundaries (optional, can be in separate file).

### Scene Data (CSV file)
Scene boundaries with columns containing start and end frame information. The exact format is parsed automatically by the framework.

## Output Files

### Generated Annotations
- `video_id_annotation.npz`: Contains all generated annotations for different body parts.

### ELAN Files
- `video_id-start_time-end_time.eaf`: Main ELAN annotation file.
- `video_id-start_time-end_time.pfsx`: ELAN preferences file.
- `video_id-start_time-end_time_tsconf.xml`: Timeseries configuration.
- `csvs/video_id-start_time-end_time_timeseries.csv`: Timeseries data in CSV format.

## Annotation Types

The framework generates several types of annotations:

### Spatial Annotations
- **Wrist positions**: Relative to body landmarks (Left, Right, Centre, Up, Down).
- **Finger positions**: Based on joint angles and orientations.
- **Spatial zones**: Combined X-Y coordinate regions.

### Movement Annotations  
- **Direction**: Movement direction changes (Left, Right, Up, Down).
- **Speed**: Velocity-based annotations with customizable thresholds.
- **Distance**: Interval-based distance measurements.

### Physiological Annotations
- **Eyelid states**: Blink detection and eye closure analysis.
- **Eyebrow positions**: Vertical positioning relative to eyes (placeholder).

## Project Structure

```
├── configs/
│   ├── pseudolabels_config.yaml    # Annotation tier configuration
│   └── elan_config.yaml            # ELAN output configuration
├── annotation/                     # Annotation generation modules
├── elan/                          # ELAN file handling
├── io_helpers.py                  # Input/output utilities
├── pseudolabels.py               # Annotation generation script
├── create_tiers.py               # ELAN file creation script
└── run_analysis.sh               # Main execution script
```

## Contributing

This framework was developed in close dialogue between computer vision scholars and linguists. Contributions should maintain compatibility with standard linguistic research tools (ELAN, CQPweb).

## Citation

TBA.

