SetFactory("OpenCASCADE");
lc = 0.25;

Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Circle(5) = {0.5, 0.5, 0, 0.25, 0, Pi};
Circle(6) = {0.5, 0.5, 0, 0.25, Pi, 2*Pi};

Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {5,6};

Plane Surface(1) = {1,2};

Physical Surface(1) = {1};

Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
//+


