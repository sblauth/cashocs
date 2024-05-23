SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1, 0};
Ellipse(5) = {0.5, 0.5, 0, 0.3, 0.15, 0, 2*Pi};

Rotate {{0.0, 0.0, 1}, {0.5, 0.5, 0}, 30.0*Pi/180.0} {
  Curve{5}; 
}
Curve Loop(2) = {5};
Plane Surface(2) = {2};

BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; }

Physical Surface(1) = {1};
Physical Surface(2) = {2};

Physical Curve(1) = {6};
Physical Curve(2) = {9};
Physical Curve(3) = {7};
Physical Curve(4) = {8};

lc = 1.5e-2;

Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].EdgesList = {1};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc;
Field[2].LcMax = lc;
Field[2].DistMin = 0.0;
Field[2].DistMax = 0.0;

Background Field = 2;


