using MATLAB
using LinearAlgebra
using PyPlot

# cd("Documents/MartinTsaiExtHJB-master/version3")

include("./sphere_utils.jl")
include("./closest_point_mappings.jl")
include("./cp_derivatives.jl")
using .SphereUtils
using .CpMaps
using .DPG


mf= MatFile("sphere_samplings.mat")
Q1=get_variable(mf, "V400")
Q = zeros(3,400)
transpose!(Q,Q1)
point_cloud=.5*Q#./.4
#Q=2*Q.-1
close(mf)
eps_factor = 4
radius = .5
x=0.5
y=0.0
z=sqrt(radius^2-(x)^2-(y)^2)
N=101
n1=N
n2=N
n3=N

dx = 2.0/(n1-1)
dy = 2.0/(n2-1)
dz = 2.0/(n3-1)
ep =eps_factor*dx #0.08#  #For this code we assume uniform grid, but if the
#grid is not uniform we should declare an epsilon in each direction
U,V,W = CpMaps.compute_cp_mapping(n1,n2,n3,point_cloud,8*ep)
X=U[:]
X=X[X.<1200.0]
Y=V[:]
Y=Y[Y.<1000.0]
Z=W[:]
Z=Z[Z.<1000.0]

fig = figure(4)
plot3D(X,Y,Z)
