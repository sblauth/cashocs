lc = 7.5e-2;

Point(1) = {0, 0, 0, lc};
Point(2) = {1.5, 0, 0, lc};
Point(3) = {4, 0, 0, lc};
Point(4) = {4.75, 0.5, 0, lc};
Point(5) = {5.4375, 1.625, 0, lc};
Point(6) = {6.125, 2.75, 0, lc};
Point(7) = {7.5, 5, 0, lc};
Point(8) = {12, 5, 0, lc};
Point(9) = {15, 5, 0, lc};
Point(10) = {15, 6, 0, lc};
Point(11) = {12, 6, 0, lc};
Point(12) = {9.75, 6, 0, lc};
Point(13) = {7.5, 6, 0, lc};
Point(14) = {6.75, 5.5, 0, lc};
Point(15) = {5.375, 3.25, 0, lc};
Point(16) = {4, 1, 0, lc};
Point(17) = {1.5, 1, 0, lc};
Point(18) = {0, 1, 0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,15};
Line(15) = {15,16};
Line(16) = {16,17};
Line(17) = {17,18};
Line(18) = {18, 1};

Line Loop(1) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};

Plane Surface(1) = {1};

Physical Surface(1) = {1};
Physical Line(1) = {18};
Physical Line(2) = {1,8,10,17};
Physical Line(3) = {9};
Physical Line(4) = {2,3,4,5,6,7,11,12,13,14,15,16};


Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].NodesList = {16, 7};
Field[1].EdgesList = {4,7,12,15};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/5;
Field[2].LcMax = lc;
Field[2].DistMin = 1e-2;
Field[2].DistMax = 1e0;

Background Field = 2;

Mesh.CharacteristicLengthExtendFromBoundary= 0;
