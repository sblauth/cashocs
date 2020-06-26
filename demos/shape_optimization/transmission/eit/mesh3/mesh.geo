lc = 3.5e-2;
radius = 0.2;
center_x = 0.5;
center_y = 0.5;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};

Point(5) = {center_x - radius, center_y - radius, 0, lc};
Point(6) = {center_x + radius, center_y - radius, 0, lc};
Point(7) = {center_x + radius, center_y + radius, 0, lc};
Point(8) = {center_x - radius, center_y + radius, 0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,5};


Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8};

Plane Surface(1) = {1,2};
Plane Surface(2) = {2};

Physical Surface(1) = {1};
Physical Surface(2) = {2};

Physical Line(1) = {1};
Physical Line(2) = {3};
Physical Line(3) = {4};
Physical Line(4) = {2};


Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].EdgesList = {5,6,7,8};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/5;
Field[2].LcMax = lc;
Field[2].DistMin = 1e-2;
Field[2].DistMax = 3e-1;

Background Field = 2;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
