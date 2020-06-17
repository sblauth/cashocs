lc = 1.0/128;
radius = 0.1;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};

Point(5) = {0.4, 0.6, 0, lc};
Point(6) = {0.4 + radius, 0.6, 0, lc};
Point(7) = {0.4, 0.6 + radius, 0, lc};
Point(8) = {0.4 - radius, 0.6, 0, lc};
Point(9) = {0.4, 0.6 - radius, 0, lc};

Point(10) = {0.55, 0.2, 0.0, lc};
Point(11) = {0.55 + radius, 0.2, 0.0, lc};
Point(12) = {0.55, 0.2 + radius, 0.0, lc};
Point(13) = {0.55 - radius, 0.2, 0.0, lc};
Point(14) = {0.55, 0.2 - radius, 0.0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Circle(9) = {11, 10, 12};
Circle(10) = {12, 10, 13};
Circle(11) = {13, 10, 14};
Circle(12) = {14, 10, 11};


Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8};
Line Loop(3) = {9,10,11,12};

Plane Surface(1) = {1,2,3};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Surface(1) = {1};
Physical Surface(2) = {2,3};

Physical Line(1) = {1};
Physical Line(2) = {3};
Physical Line(3) = {4};
Physical Line(4) = {2};
