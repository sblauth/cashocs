lc = 3.5e-2;
radius = 0.2;
center_x = 0.5;
center_y = 0.5;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};


Point(5) = {0.5, 0.7, 0, lc};
Point(6) = {0.6, 0.675, 0, lc};
Point(7) = {0.6, 0.6, 0, lc};
Point(8) = {0.65, 0.5, 0, lc};
Point(9) = {0.675, 0.45, 0, lc};
Point(10) = {0.65, 0.35, 0, lc};
Point(11) = {0.5, 0.3, 0, lc};
Point(12) = {0.35, 0.35, 0, lc};
Point(13) = {0.325, 0.45, 0, lc};
Point(14) = {0.35, 0.5, 0, lc};
Point(15) = {0.4, 0.6, 0, lc};
Point(16) = {0.4, 0.675, 0, lc};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

BSpline(5) = {5,6,7,8,9,10,11,12,13,14,15,16,5};


Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5};

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
