Merge 'mesh_1_pre_remesh.msh';
CreateGeometry;

lc = 5e-2;
Field[1] = Distance;
Field[1].NNodesByEdge = 1000;
Field[1].NodesList = {2};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].DistMin = 1e-1;
Field[2].DistMax = 5e-1;
Field[2].LcMin = lc / 10;
Field[2].LcMax = lc;
Background Field = 2;
