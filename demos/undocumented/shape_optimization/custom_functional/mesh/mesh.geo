lc = 7.5e-2;

SetFactory("OpenCASCADE");
Disk(1) = {0, 0, 0, 1, 1};

Mesh.MeshSizeMax = lc;
Mesh.MeshSizeMin = 0.0;

Physical Curve(1) = {1};
Physical Surface(1) = {1};
