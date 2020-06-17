lc = 1.0/128;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};

//center ellipse 1
Point(5) = {0.6, 0.7, 0.0, lc};
//ellipse 1
a_minor = 0.08;
a_major = 0.145;
Point(6) = {0.6 + a_major, 0.7, 0.0, lc};
Point(7) = {0.6, 0.7 + a_minor, 0.0, lc};
Point(8) = {0.6 - a_major, 0.7, 0.0, lc};
Point(9) = {0.6, 0.7 - a_minor, 0.0, lc};

//center ellipse 2
Point(10) = {0.4, 0.3, 0.0, lc};
//ellipse 2
Point(11) = {0.4 + a_minor, 0.3, 0.0, lc};
Point(12) = {0.4, 0.3 + a_major, 0.0, lc};
Point(13) = {0.4 - a_minor, 0.3, 0.0, lc};
Point(14) = {0.4, 0.3 - a_major, 0.0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Ellipse(5) = {6,5,6,7};
Ellipse(6) = {7,5,6,8};
Ellipse(7) = {8,5,6,9};
Ellipse(8) = {9,5,6,6};

Ellipse(9) = {11, 10, 11, 12};
Ellipse(10) = {12, 10, 11, 13};
Ellipse(11) = {13, 10, 11, 14};
Ellipse(12) = {14, 10, 11, 11};


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
