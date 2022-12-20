Merge './fine/mesh.msh';
CreateGeometry;

lc = 0.5e-1;
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


