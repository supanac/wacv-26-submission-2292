def merge_label(label: str) -> str:
    """
    Merge and simplify compound spatial labels into consolidated directional labels.
    
    Args:
        label (str): Space-separated label string with X and Y directional components.
    
    Returns:
        str: Simplified directional label string, or None if no match found.
    """
    label_x, label_y = label.split(" ")
    if (
        (label_x == "Right" and label_y == "CentreUp")
        or
        (label_x == "CentreRight" and label_y == "Up")
        or
        (label_x == "Right" and label_y == "Up")
    ):
        return "Right Up"
    if (
        (label_x == "Left" and label_y == "CentreUp")
        or
        (label_x == "CentreLeft" and label_y == "Up")
        or
        (label_x == "Left" and label_y == "Up")
    ):
        return "Left Up"
    if (
        (label_x == "Right" and label_y == "CentreDown")
        or
        (label_x == "CentreRight" and label_y == "Down")
        or
        (label_x == "Right" and label_y == "Down")
    ):
        return "Right Down"
    if (
        (label_x == "Left" and label_y == "CentreDown")
        or
        (label_x == "CentreLeft" and label_y == "Down")
        or
        (label_x == "Left" and label_y == "Down")
    ):
        return "Left Down"
    if label_x == "CentreX" and label_y == "Up":
        return "Up"
    if label_x == "CentreRight" and label_y == "CentreUp":
        return "Centre Right Up"
    if label_x == "CentreX" and label_y == "CentreUp":
        return "Centre Up"
    if label_x == "CentreLeft" and label_y == "CentreUp":
        return "Centre Left Up"
    if label_x == "Right" and label_y == "CentreY":
        return "Right"
    if label_x == "CentreRight" and label_y == "CentreY":
        return "Centre Right"
    if label_x == "CentreX" and label_y == "CentreY":
        return "Centre"
    if label_x == "CentreLeft" and label_y == "CentreY":
        return "Centre Left"
    if label_x == "Left" and label_y == "CentreY":
        return "Left"
    if label_x == "CentreRight" and label_y == "CentreDown":
        return "Centre Right Down"
    if label_x == "CentreX" and label_y == "CentreDown":
        return "Centre Down"
    if label_x == "CentreLeft" and label_y == "CentreDown":
        return "Centre Left Down"
    if label_x == "CentreX" and label_y == "Down":
        return "Down"