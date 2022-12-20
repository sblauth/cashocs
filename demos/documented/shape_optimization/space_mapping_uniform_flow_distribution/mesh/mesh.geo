SetFactory("OpenCASCADE");
lc = 1e-1;

d_len = 6;
c_len = 3.0;
c_width = 0.75;
d_width = 1;
r_bend = 0.66;
c_dist = 1;

Point(1) = {0, 0, 0, lc};
Point(2) = {d_len - r_bend, 0, 0, lc};
Point(3) = {d_len - r_bend, -r_bend, 0, lc};
Point(4) = {d_len, -r_bend, 0, lc};
Point(5) = {d_len, -d_width, 0, lc};
Point(6) = {d_len, -d_width-c_len/2, 0, lc};
Point(7) = {d_len, -d_width-c_len, 0, lc};
Point(8) = {d_len - c_width, -d_width-c_len, 0, lc};
Point(9) = {d_len - c_width, -d_width-c_len/2, 0, lc};
Point(10) = {d_len - c_width, -d_width, 0, lc};
Point(11) = {d_len - c_width - c_dist, -d_width, 0, lc};
Point(12) = {d_len - c_width - c_dist, -d_width-c_len/2, 0, lc};
Point(13) = {d_len - c_width - c_dist, -d_width-c_len, 0, lc};
Point(14) = {d_len - 2*c_width - c_dist, -d_width-c_len, 0, lc};
Point(15) = {d_len - 2*c_width - c_dist, -d_width-c_len/2, 0, lc};
Point(16) = {d_len - 2*c_width - c_dist, -d_width, 0, lc};
Point(17) = {d_len - 2*c_width - 2*c_dist, -d_width, 0, lc};
Point(18) = {d_len - 2*c_width - 2*c_dist, -d_width-c_len/2, 0, lc};
Point(19) = {d_len - 2*c_width - 2*c_dist, -d_width-c_len, 0, lc};
Point(20) = {d_len - 3*c_width - 2*c_dist, -d_width-c_len, 0, lc};
Point(21) = {d_len - 3*c_width - 2*c_dist, -d_width-c_len/2, 0, lc};
Point(22) = {d_len - 3*c_width - 2*c_dist, -d_width, 0, lc};
Point(23) = {0, -d_width, 0, lc};

Line(1) = {1,2};
Circle(2) = {2,3,4};
Line(3) = {4,5};
Line(4) = {5,6};
Line(5) = {6,7};
Line(6) = {7,8};
Line(7) = {8,9};
Line(8) = {9,10};
Line(9) = {10,11};
Line(10) = {11,12};
Line(11) = {12,13};
Line(12) = {13,14};
Line(13) = {14,15};
Line(14) = {15,16};
Line(15) = {16,17};
Line(16) = {17,18};
Line(17) = {18,19};
Line(18) = {19,20};
Line(19) = {20,21};
Line(20) = {21,22};
Line(21) = {22,23};
Line(22) = {23,1};

Line(23) = {5,10};
Line(24) = {11,16};
Line(25) = {17,22};


Line Loop(1) = {1,2,3,23,9,24,15,25,21,22};
Line Loop(2) = {4,5,6,7,8,-23};
Line Loop(3) = {10,11,12,13,14,-24};
Line Loop(4) = {16,17,18,19,20,-25};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

Physical Surface(10) = {1};
Physical Surface(20) = {2,3,4};


Physical Curve(1) = {22};
Physical Curve(2) = {1, 2, 3, 9, 15, 21};
Physical Curve(3) = {20, 16, 14, 10, 8, 4};
Physical Curve(4) = {19, 17, 13, 11, 7, 5};
Physical Curve(5) = {18};
Physical Curve(6) = {12};
Physical Curve(7) = {6};


Field[1] = Distance;
Field[1].NNodesByEdge = 5000;
Field[1].EdgesList = {1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/5;
Field[2].LcMax = lc;
Field[2].DistMin = 1e-2;
Field[2].DistMax = 2.5e-1;

Field[3] = Min;
Field[3].FieldsList = {2};

Background Field = 3;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Physical Curve(100) = {25, 24, 23};

Mesh.SaveElementTagType = 2;
