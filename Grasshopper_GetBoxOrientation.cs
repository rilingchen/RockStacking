// Grasshopper C# Script: get orientation from BoxObject (DCT)
// Input: BoxObj. Outputs: P, C, X, Y, Z. Edit the path below if your Libraries folder is elsewhere.
System.Reflection.Assembly.LoadFrom("/Users/r2d2/Library/Application Support/McNeel/Rhinoceros/8.0/Plug-ins/Grasshopper (b45a29b1-4343-4035-989e-044e8580d9cf)/Libraries/DigitalCircularityToolkit.gha");

var dct = boxObj as DigitalCircularityToolkit.Objects.BoxObject;
if (dct == null) {
  Print("Input is not a BoxObject.");
  return;
}
P = dct.Plane;           // orientation plane (center + X/Y/Z axes)
C = dct.Plane.Origin;    // center
X = dct.Plane.XAxis;     // length direction (PCA1)
Y = dct.Plane.YAxis;     // width direction (PCA2)
Z = dct.Plane.ZAxis;     // height direction (PCA3)
