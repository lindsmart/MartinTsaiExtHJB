# MartinTsaiExtHJB

This is the Julia code used to compute the numerical solutions in the paper "Equivalent extensions of Hamilton-Jacobi-Bellman equations on hypersurfaces" by Lindsay Martin and Richard Tsai.

To use this code one needs the following Julia packages:

MATLAB
NearestNeighbors
LinearAlgebra
Interpolations

The test scripts are:

Spheretests.jl -used to compute the convergence tests in the paper (Example 1)
torustest.jl - distance function on the torus (Example 2)
torussort.jl - sort the point cloud into "level belts" (Example 2)
Bunnytest.jl - distance function on the Stanford bunny (Example 3)
anisotropictest.jl -anisotropic HJB equation with curvature based speed function (Example 3)

The other Julia files are modules included at the beginning of each of the test files.

Included is also the point cloud data used in the paper. The point clouds are stored in Matlab data files and read into Julia.
