lc = 5e-2;

Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {0, 1, 0, lc};
Point(4) = {-1, 0, 0, lc};
Point(5) = {0, -1, 0, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Line Loop(1) = {1,2,3,4};

Surface(1) = {1};

Physical Surface(1) = {1};

Physical Line(1) = {1,2,3,4};

Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].NodesList = {2};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].DistMin = 1e-1;
Field[2].DistMax = 5e-1;
Field[2].LcMin = lc / 5;
Field[2].LcMax = lc;

Background Field = 2;
