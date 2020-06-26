lc = 3.5e-1;

Point(1) = {-3, -2, -2, lc};
Point(2) = {6, -2, -2, lc};
Point(3) = {6, 2, -2, lc};
Point(4) = {-3, 2, -2, lc};
Point(5) = {-3, -2, 2, lc};
Point(6) = {6, -2, 2, lc};
Point(7) = {6, 2, 2, lc};
Point(8) = {-3, 2, 2, lc};

Point(9) = {0, 0, 0, lc};
Point(10) = {0, 0.5, 0, lc};
Point(11) = {0, -0.5, 0, lc};
Point(12) = {0.5, 0, 0, lc};
Point(13) = {-0.5, 0, 0, lc};
Point(14) = {0, 0, 0.5, lc};
Point(15) = {0, 0, -0.5, lc};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,5};

Line(9) = {1,5};
Line(10) = {8,4};

Line(11) = {6,2};
Line(12) = {3,7};

Circle(13) = {10, 9, 12};
Circle(14) = {10, 9, 14};
Circle(15) = {10, 9, 13};
Circle(16) = {10, 9, 15};

Circle(17) = {12, 9, 11};
Circle(18) = {14, 9, 11};
Circle(19) = {13, 9, 11};
Circle(20) = {15, 9, 11};

Circle(21) = {12, 9, 14};
Circle(22) = {14, 9, 13};
Circle(23) = {13, 9, 15};
Circle(24) = {15, 9 ,12};


Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8};
Line Loop(3) = {10,4,9,-8};
Line Loop(4) = {11,2,12,-6};
Line Loop(5) = {9,5,11,-1};
Line Loop(6) = {12,7,10,-3};

Line Loop(7) = {13, 21, -14};
Line Loop(8) = {14, 22, -15};
Line Loop(9) = {15, 23, -16};
Line Loop(10) = {16, 24, -13};
Line Loop(11) = {-17, 21, 18};
Line Loop(12) = {-18, 22, 19};
Line Loop(13) = {-19, 23, 20};
Line Loop(14) = {-20, 24, 17};


Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};

Surface(7) = {7};
Surface(8) = {8};
Surface(9) = {9};
Surface(10) = {10};
Surface(11) = {11};
Surface(12) = {12};
Surface(13) = {13};
Surface(14) = {14};

Surface Loop(1) = {1,2,3,4,5,6};
Surface Loop(2) = {7,8,9,10,11,12,13,14};

Volume(1) = {1, 2};

Physical Surface(1) = {3};
Physical Surface(2) = {2, 5, 1, 6};
Physical Surface(3) = {4};
Physical Surface(4) = {7, 9, 10, 14, 11, 12, 13, 8};

Physical Volume(1) = {1};


Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].EdgesList = {13,14,15,16,17,18,19,20,21,22,23,24};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/5;
Field[2].LcMax = lc;
Field[2].DistMin = 2.5e-2;
Field[2].DistMax = 5e-2;

Background Field = 2;
