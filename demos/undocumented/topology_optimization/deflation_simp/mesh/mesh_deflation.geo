lc = 2e-2;
radius = 0.05;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.5, 0.0, 0.0, lc};
Point(3) = {1.5, 1.0, 0.0, lc};
Point(4) = {0.0, 1.0, 0.0, lc};

Point(5) = {0.0, 1./6, 0.0, lc};
Point(6) = {0.0, 2./6, 0.0, lc};
Point(7) = {0.0, 4./6, 0.0, lc};
Point(8) = {0.0, 5./6, 0.0, lc};

Point(9) = {1.5, 1./6, 0.0, lc};
Point(10) = {1.5, 2./6, 0.0, lc};
Point(11) = {1.5, 4./6, 0.0, lc};
Point(12) = {1.5, 5./6, 0.0, lc};

Point(20) = {0.5, 1./3, 0.0, lc};
Point(21) = {0.5 - radius, 1./3, 0.0, lc};
Point(22) = {0.5, 1./3 - radius, 0.0, lc};
Point(23) = {0.5 + radius, 1./3, 0.0, lc};
Point(24) = {0.5, 1./3 + radius, 0.0, lc};

Point(30) = {0.5, 2./3, 0.0, lc};
Point(31) = {0.5 - radius, 2./3, 0.0, lc};
Point(32) = {0.5, 2./3 - radius, 0.0, lc};
Point(33) = {0.5 + radius, 2./3, 0.0, lc};
Point(34) = {0.5, 2./3 + radius, 0.0, lc};

Point(40) = {1.0, 1./4, 0.0, lc};
Point(41) = {1.0 - radius, 1./4, 0.0, lc};
Point(42) = {1.0, 1./4 - radius, 0.0, lc};
Point(43) = {1.0 + radius, 1./4, 0.0, lc};
Point(44) = {1.0, 1./4 + radius, 0.0, lc};

Point(50) = {1.0, 1./2, 0.0, lc};
Point(51) = {1.0 - radius, 1./2, 0.0, lc};
Point(52) = {1.0, 1./2 - radius, 0.0, lc};
Point(53) = {1.0 + radius, 1./2, 0.0, lc};
Point(54) = {1.0, 1./2 + radius, 0.0, lc};

Point(60) = {1.0, 3./4, 0.0, lc};
Point(61) = {1.0 - radius, 3./4, 0.0, lc};
Point(62) = {1.0, 3./4 - radius, 0.0, lc};
Point(63) = {1.0 + radius, 3./4, 0.0, lc};
Point(64) = {1.0, 3./4 + radius, 0.0, lc};

Line(1) = {1,2};
Line(2) = {2,9};
Line(3) = {9,10};
Line(4) = {10,11};
Line(5) = {11,12};
Line(6) = {12,3};
Line(7) = {3,4};
Line(8) = {4,8};
Line(9) = {8,7};
Line(10) = {7,6};
Line(11) = {6,5};
Line(12) = {5,1};

Circle(20) = {21,20,22};
Circle(21) = {22,20,23};
Circle(22) = {23,20,24};
Circle(23) = {24,20,21};

Circle(30) = {31,30,32};
Circle(31) = {32,30,33};
Circle(32) = {33,30,34};
Circle(33) = {34,30,31};

Circle(40) = {41,40,42};
Circle(41) = {42,40,43};
Circle(42) = {43,40,44};
Circle(43) = {44,40,41};

Circle(50) = {51,50,52};
Circle(51) = {52,50,53};
Circle(52) = {53,50,54};
Circle(53) = {54,50,51};

Circle(60) = {61,60,62};
Circle(61) = {62,60,63};
Circle(62) = {63,60,64};
Circle(63) = {64,60,61};

Line Loop(1) = {1,2,3,4,5,6,7,8,9,10,11,12};
Line Loop(2) = {20,21,22,23};
Line Loop(3) = {30,31,32,33};
Line Loop(4) = {40,41,42,43};
Line Loop(5) = {50,51,52,53};
Line Loop(6) = {60,61,62,63};

Plane Surface(1) = {1, 2, 3, 4, 5, 6};

Physical Surface(1) = {1};

Physical Line(1) = {1,2,4,6,7,8,10,12,20,21,22,23,30,31,32,33,40,41,42,43,50,51,52,53,60,61,62,63};
Physical Line(2) = {9};
Physical Line(3) = {11};
Physical Line(4) = {5};
Physical Line(5) = {3};
