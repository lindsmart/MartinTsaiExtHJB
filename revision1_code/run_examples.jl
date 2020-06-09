using DocStringExtensions
using LinearAlgebra
using PrettyTables
using MATLAB

include("./sphere_tests.jl")
using .SphereTests




for i=1:1
#point cloud for sphere comment out lines 14-20 if not using point cloud code
mf= MatFile("sphere_samplings.mat")
point_cloud = "V400"# possibilities are: "V400_N001", "V400_N005"#,"V800","V800_N001", "V800_N005","V1600","V1600_N001", "V1600_N005"

Q1=get_variable(mf, point_cloud)
Q = zeros(3,size(Q1)[1])
transpose!(Q,Q1)
Q=.5*Q

#source point
radius = .5
x=0.5
y=0.0
z=sqrt(radius^2-(x)^2-(y)^2)

#viscosity parameter if enforcing
sig = 1.0

num_grid_sizes=1
global rows = zeros(num_grid_sizes,8)

i=1
orderL1 = 0.0
orderLinf = 0.0
orderAvg= 0.0
L1 = 0.0
Linf = 0.0
Nprev = 0.0
avgError =0.0

columns = ["N" "#points" "L1" "order" "Linf" "order " "iters" "time"]
grid_sizes = [101,201,301,401]
grid_sizes= grid_sizes[1:num_grid_sizes]

for N in grid_sizes

    L1prev=L1
    Linfprev=Linf
    avgErrorprev = avgError
     #uncomment the test you're trying to run and comment out the others
      time = @elapsed L1, Linf, avgError, num_iters, num_pts = SphereTests.main_exact_sphere_high_order(N,N,N,radius,[x,y,z], 11, sig,sig,sig, 400)
     #time =  @elapsed L1, Linf, avgError, num_iters, num_pts = SphereTests.main_point_cloud_sphere_high_order(Q,N,N,N,radius,[x,y,z],7, sig,sig,sig, 200)
     #time =  @elapsed L1, Linf, avgError, num_iters, num_pts = SphereTests.main_point_cloud_sphere(Q,N,N,N,radius,[x,y,z],7, sig,sig,sig, 200)

    # time =  @elapsed L1, Linf, num_iters, num_pts = SphereTests.main_exact_sphere(N,N,N,radius,[x,y,z], 4, sig,sig,sig, 200)

    if i > 1
        orderL1 = log(L1/L1prev)/log(Nprev/N)
        orderLinf =log(Linf/Linfprev)/log(Nprev/N)
        orderAvg = log(avgError/avgErrorprev)/log(Nprev/N)
    end
    Nprev = N
    rows[i,:] = [ N num_pts L1 orderL1 Linf orderLinf num_iters time ]
    i = i+1
    pretty_table(rows, columns)

end


end
