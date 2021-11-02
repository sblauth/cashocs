lc = 2e-1;

length = 3.0;
width = 2.0;

Point(1) = {-length, -width, 0, lc};
Point(2) = {length, -width, 0, lc};
Point(3) = {length, width, 0, lc};
Point(4) = {-length, width, 0, lc};

Point(5) = {0.5, 0, 0, lc};
Point(6) = {0, 0.5, 0, lc};
Point(7) = {-0.5, 0, 0, lc};
Point(8) = {0, -0.5, 0, lc};
Point(9) = {0, 0, 0, lc};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Circle(5) = {5,9,6};
Circle(6) = {6,9,7};
Circle(7) = {7,9,8};
Circle(8) = {8,9,5};

Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8};

Plane Surface(1) = {1,2};


Physical Surface(1) = {1};

Physical Line(1) = {4};
Physical Line(2) = {1,3};
Physical Line(3) = {2};
Physical Line(4) = {5,6,7,8};


Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].EdgesList = {5,6,7,8};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/30;
Field[2].LcMax = lc;
Field[2].DistMin = 0;
Field[2].DistMax = 1.25;
Field[2].Sigmoid = 0;

Field[3] = Distance;
Field[3].NNodesByEdge = 1000;
Field[3].EdgesList = {1,2,3,4};
Field[4] = Threshold;
Field[4].IField = 3;
Field[4].LcMin = lc/5;
Field[4].LcMax = lc;
Field[4].DistMin = 0;
Field[4].DistMax = 0.5;
Field[4].Sigmoid = 0;

Field[5] = Min;
Field[5].FieldsList = {2,4};

Background Field = 5;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
