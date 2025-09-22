# Description

This reposiitory contains code that combines ASD output 
with OpenPose json files and produces a numpy file with
active speaker's coordinates.

---
# Run
```
python extract_active_speakers.py --video-id [VIDEO-ID] --source-fps [SOURCE-FPS] --video-width [VIDEO-WIDTH]
```

---
# Output
The script scores 
1. `pose_keypoints_2d`: Raw, i.e. _non-normalised_ **but** _interpolated_ values of keypoints coordinates.
2. `pose_keypoints_2d_norm`: Normalised and interpolated version of the raw coordinates.
3. `track_indices`: A vector that stores a track number for each frame. 
4. `confidence_scores`: Confidence scores for each keypoints estimated by OpenPose.

**Nota bene**: The rule of thumb for finding an active speaker is `score > 0`. 
Therefore we might encounter a situation in which we use frames from
`tracks.pckl` as indices for `pose_keypoints_2d_norm` but have some
zeros either at the beginning, or in the middle, or at the end.
These zeros are due to negative scores from `scores.pckl`.