# Grasshopper Python: get orientation from BoxObject (DCT)
# 1. Add a "Python" or "Script" component
# 2. Input: BoxObj (item from BoxObject DCT component)
# 3. Outputs: P, C, X, Y, Z (add 5 output params)
# 4. Wire BoxObject (DCT) BoxObj output → BoxObj input

import clr
clr.AddReference("DigitalCircularityToolkit")
from DigitalCircularityToolkit.Objects import BoxObject as DCTBoxObject

box_obj = BoxObj
if not isinstance(box_obj, DCTBoxObject):
    raise ValueError("Input is not a BoxObject.")

plane = box_obj.Plane

P = plane
C = plane.Origin
X = plane.XAxis   # length (PCA1)
Y = plane.YAxis   # width (PCA2)
Z = plane.ZAxis   # height (PCA3)
