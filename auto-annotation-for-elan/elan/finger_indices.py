def get_finger_indices(annotation_tier: str) -> int:
    finger_ind_dict = {
        "thumb": [3, 4],
        "index": [6, 8],
        "middle": [10, 12],
        "ring": [14, 16],
        "little": [18, 20]
    }
    finger_name = [
        key for key in finger_ind_dict.keys() if key in annotation_tier
        ]
    assert finger_name
    finger_ind = finger_ind_dict[finger_name[0]]
    return finger_ind
