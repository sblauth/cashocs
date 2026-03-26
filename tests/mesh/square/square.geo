SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.5, 0.5, 0};
Rectangle(2) = {0.5, 0, 0, 0.5, 0.5, 0};
Rectangle(3) = {0.5, 0.5, 0, 0.5, 0.5, 0};
Rectangle(4) = {0, 0.5, 0, 0.5, 0.5, 0};
BooleanFragments{ Surface{4}; Surface{3}; Surface{2}; Surface{1}; Delete; }{ }

Physical Surface("top_left", 1) = {4};
Physical Surface("top_right", 2) = {3};
Physical Surface("bottom_right", 3) = {2};
Physical Surface("bottom_left", 4) = {1};

Physical Curve("left", 5) = {4, 12};
Physical Curve("right", 6) = {9, 6};
Physical Curve("bottom", 7) = {8, 11};
Physical Curve("top", 8) = {7, 3};
Physical Curve("interior", 9) = {1, 2, 10, 5};
