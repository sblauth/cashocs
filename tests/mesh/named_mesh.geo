lc = 1e-1;

Point(1) = {0,0,0,lc};
Point(2) = {1,0,0,lc};
Point(3) = {1,1,0,lc};
Point(4) = {0,1,0,lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Physical Surface("volume", 1) = {1};

Physical Curve("inlet", 1) = {4};
Physical Curve("wall", 2) = {1,3};
Physical Curve("outlet", 3) = {2};
