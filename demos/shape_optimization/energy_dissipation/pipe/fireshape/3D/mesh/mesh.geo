// Gmsh project created on Tue Jan 22 11:40:52 2019
lc = 1.5e-1;
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 0.5, 0, 2*Pi};
Line Loop(2) = {1};
Plane Surface(1) = {2};
Point(5) = {0, 0.0,  0.00, lc};
Point(6) = {0, 0.0,  2.00, lc}; //first spline point
Point(7) = {0, 0.0,  3.00, lc};
Point(8) = {0, 0.0,  4.00, lc};
Point(9) = {0, 0.2,  4.75, lc};
Point(10) = {0, 5.0,  6.00, lc};
Point(11) = {0, 5.0, 10.00, lc};
Point(12) = {0, 5.0, 12.00, lc}; //last spline point
Point(13) = {0, 5.0, 15.00, lc};

Bezier(10) = {6, 7, 8, 9, 10, 11, 12};
Line(15) = {5, 6};
Line(16) = {12, 13};
Wire(1) = {15, 10, 16};
Extrude { Surface{1}; } Using Wire {1}
Delete{ Surface{1}; }


Physical Surface(1) = {2};
Physical Surface(2) = {3, 5};
Physical Surface(3) = {6};
Physical Surface(4) = {4};

Physical Volume(1) = {1};

Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].EdgesList = {15};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc;
Field[2].LcMax = lc;
Field[2].DistMin = 1;
Field[2].DistMax = 1;

Background Field = 2;
