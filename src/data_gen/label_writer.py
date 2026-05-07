def normalize_box(x, y, w, h, img_width, img_height):
    """
    Convert a bounding box from pixel coordinates to YOLO format.

    YOLO requires all values to be normalized to [0, 1] relative
    to the image dimensions.

    Parameters
    ----------
    x : int | float
        Left edge of the bounding box in pixels (top-left origin).
    y : int | float
        Top edge of the bounding box in pixels (top-left origin).
    w : int | float
        Width of the bounding box in pixels.
    h : int | float
        Height of the bounding box in pixels.
    img_width : int | float
        Total width of the image in pixels.
    img_height : int | float
        Total height of the image in pixels.

    Returns
    -------
    dict
        {"x_center", "y_center", "width", "height"} — all in [0, 1].

    Formula (derived from first principles, 2026-05-03)
    -------
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width    = w / img_width
        height   = h / img_height
    """
    return {
        "x_center": (x + w / 2) / img_width,
        "y_center": (y + h / 2) / img_height,
        "width":    w / img_width,
        "height":   h / img_height,
    }


def write_label_file(path, annotations, img_width, img_height):
    """
    Write a YOLO-format label file to disk.

    Each annotation is written as one line:
        <class_id> <x_center> <y_center> <width> <height>

    All spatial values are normalized to [0, 1] via normalize_box.

    Parameters
    ----------
    path : str
        Full file path for the output .txt file.
    annotations : list[dict]
        Each dict must have keys: class_id, x, y, w, h (pixel coords).
    img_width : int | float
        Image width in pixels (used for normalization).
    img_height : int | float
        Image height in pixels (used for normalization).
    """
    with open(path, "w") as f:
        for ann in annotations:
            box = normalize_box(
                x=ann["x"], y=ann["y"],
                w=ann["w"], h=ann["h"],
                img_width=img_width, img_height=img_height,
            )
            line = (
                f"{ann['class_id']} "
                f"{box['x_center']} "
                f"{box['y_center']} "
                f"{box['width']} "
                f"{box['height']}"
            )
            f.write(line + "\n")
