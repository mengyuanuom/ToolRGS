"""Image-bounded integer rectangle IoU, with antipodal grasp angles.

Coordinates are (x,y); image shape is (height,width). Scanline intervals count
the same integer pixels as skimage.draw.polygon(y,x,shape), without allocating
an image-sized mask for every pair. No continuous-area approximation is used.
"""
import cv2
import numpy as np

GEOMETRY_VERSION = 'raster_xy_periodic180_v2'


def angle_distance(a, b):
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def rectangle_spans(rect, shape):
    r = np.asarray(rect, dtype=np.float64).reshape(-1)
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError('Image shape must be positive (height,width)')
    if r.size < 5 or not np.isfinite(r[:5]).all():
        raise ValueError('Non-finite or incomplete grasp rectangle')
    if r[2] <= 0 or r[3] <= 0:
        return (0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0)
    box = cv2.boxPoints(((float(r[0]), float(r[1])),
                         (float(r[2]), float(r[3])), -float(r[4]))).astype(np.int64)
    y0, y1 = max(0, int(box[:, 1].min())), min(h-1, int(box[:, 1].max()))
    if y1 < y0:
        return (0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0)
    ys = np.arange(y0, y1+1)
    low = np.full(len(ys), np.inf)
    high = np.full(len(ys), -np.inf)
    for p, q in zip(box, np.roll(box, -1, axis=0)):
        valid = (ys >= min(p[1], q[1])) & (ys <= max(p[1], q[1]))
        if p[1] == q[1]:
            low[valid] = np.minimum(low[valid], min(p[0], q[0]))
            high[valid] = np.maximum(high[valid], max(p[0], q[0]))
        else:
            xs = p[0] + (ys[valid]-p[1]) * (q[0]-p[0]) / float(q[1]-p[1])
            low[valid] = np.minimum(low[valid], xs)
            high[valid] = np.maximum(high[valid], xs)
    left = np.maximum(0, np.ceil(low-1e-9).astype(np.int64))
    right = np.minimum(w-1, np.floor(high+1e-9).astype(np.int64))
    return y0, left, right, int(np.maximum(0, right-left+1).sum())


def spans_iou(a, b):
    y0 = max(a[0], b[0])
    y1 = min(a[0]+len(a[1]), b[0]+len(b[1]))
    intersection = 0
    if y1 > y0:
        aa = slice(y0-a[0], y1-a[0])
        bb = slice(y0-b[0], y1-b[0])
        intersection = int(np.maximum(0, np.minimum(a[2][aa], b[2][bb])
                                      -np.maximum(a[1][aa], b[1][bb])+1).sum())
    union = a[3]+b[3]-intersection
    return intersection/union if union > 0 else 0.0


def rectangle_iou(prediction, target, shape=(720,1280), angle_threshold=30):
    if not np.isfinite(np.asarray(prediction)[:5]).all() or not np.isfinite(np.asarray(target)[:5]).all():
        raise ValueError('Non-finite grasp geometry')
    if angle_distance(prediction[4], target[4]) > angle_threshold:
        return 0.0
    return spans_iou(rectangle_spans(prediction,shape), rectangle_spans(target,shape))


def grasp_matches(predictions, targets, target_width_cap=100., target_height=20., shape=(720,1280)):
    p = np.asarray(predictions, dtype=np.float64).reshape(-1,5)
    g = np.asarray(targets, dtype=np.float64).reshape(-1,6).copy()
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Non-finite grasp geometry')
    if target_height is not None:
        g[:,3] = float(target_height)
    if target_width_cap is not None:
        g[:,2] = np.clip(g[:,2], 0., float(target_width_cap))
    gs = [rectangle_spans(t,shape) for t in g]
    result=[]
    for i,r in enumerate(p):
        ps=rectangle_spans(r,shape)
        result.extend((i,spans_iou(ps,t),angle_distance(r[4],gt[4])) for t,gt in zip(gs,g))
    return result
