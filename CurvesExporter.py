#! python3
# requirements: pandas

import numpy as np

params = np.linspace(0, 1, 11)

for curve in x:
    for i in range(len(params) - 1):
        pts = []
        for t in params[i:i + 2]:
            pt = curve.PointAt(t)
            x, y = pt.X, pt.Y
            pts.append((x, y))
        print(pts)