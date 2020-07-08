lc = 3.75e-2;

Point(1) = {0, 0, 0, lc};
Point(2) = {1.5, 0, 0, lc};
Point(3) = {4, 0, 0, lc};
Point(4) = {4.75, 0.5, 0, lc};
Point(5) = {6.125, 2.75, 0, lc};
Point(6) = {7.5, 5, 0, lc};
Point(7) = {12, 5, 0, lc};
Point(8) = {15, 5, 0, lc};
Point(9) = {15, 6, 0, lc};
Point(10) = {12, 6, 0, lc};
Point(11) = {7.5, 6, 0, lc};
Point(12) = {6.75, 5.5, 0, lc};
Point(13) = {5.375, 3.25, 0, lc};
Point(14) = {4, 1, 0, lc};
Point(15) = {1.5, 1, 0, lc};
Point(16) = {0, 1, 0, lc};

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
Line(16) = {16,1};

Line Loop(1) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

Plane Surface(1) = {1};

Physical Surface(1) = {1};
Physical Line(1) = {16};
Physical Line(2) = {1,7,9,15};
Physical Line(3) = {8};
Physical Line(4) = {2,3,4,5,6,9,10,11,12,13,14};


Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].NodesList = {5, 12};
Field[1].EdgesList = {4,6,10,13};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/5;
Field[2].LcMax = lc;
Field[2].DistMin = 1e-2;
Field[2].DistMax = 1e0;

//Background Field = 2;
//Mesh.CharacteristicLengthExtendFromBoundary= 0;
