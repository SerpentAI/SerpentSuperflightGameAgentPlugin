
def expand_bounding_box(bounding_box, shape, x, y):
    y0, x0, y1, x1 = bounding_box

    y0 = y0 - y if y0 - y >= 0 else 0
    y1 = y1 + y if y1 + y <= shape[0] else shape[0]
    x0 = x0 - x if x0 - x >= 0 else 0
    x1 = x1 + x if x1 + x <= shape[1] else shape[1]

    return y0, y1, x0, x1
