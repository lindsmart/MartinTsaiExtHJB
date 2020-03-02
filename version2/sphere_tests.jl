using DocStringExtensions
using LinearAlgebra
include("./sphere_utils.jl")
include("./closest_point_mappings.jl")
using .SphereUtils
using .CpMaps
"""
This function computes the distance function on the sphere to the target point.
        $(SIGNATURES)
        - n1: The number of grid nodes in the x direction
        - n2: The number of grid nodes in the y direction
        - n3: The number of grid nodes in the z direction
        - radius: The radius of the sphere
        - target_point: The target point on the sphere
        - eps_factor: Here, epsilon = eps_factor * 1/h
        - sigma_x: x viscosity parameter
        - sigma_y: y viscosity parameter
        - sigma_z: z viscosity parameter
"""
function main_exact_sphere(n1::Int64,
                           n2::Int64,
                           n3::Int64,
                           radius::Float64,
                           target_point::Array{Float64,1},
                           eps_factor::Int64,
                           sigma_x::Float64,
                           sigma_y::Float64,
                           sigma_z::Float64,
                           niters::Int64)


        dx = 1.0/(n1-1)
        dy = 1.0/(n2-1)
        dz = 1.0/(n3-1)
        eps = eps_factor*dx # For this code we assume uniform grid, but if the
        #grid is not uniform we should declare an epsilon in each direction

        ∞ = 1000.0
        #initilize status matrix: if status[i,j,k] is true, the grid node lies
        #in the extended surface
        status=fill(false, (n1, n2,n3))
        edge=fill(false, (n1, n2,n3))
        rhs = ∞ * ones(n1,n2,n3) # right hand side matrix for sphere test

        for I in CartesianIndices(status)
            i = I[1]
            j = I[2]
            k = I[3]
            x = (i-1)*dx
            y = (j-1)*dy
            z = (k-1)*dz
            r = sqrt((x-.5)^2 +(y-.5)^2+(z-.5)^2)
            if abs(r - radius) < eps + 1e-6
                status[i,j,k] = true
                rhs[i,j,k] = 1 - (r - radius)/r
            end
        end

        #initalize edge matrix, if edge[i,j,k] is in domain and borders any point
        #not in the extended mark edge[i,j,k] as true

        for I in CartesianIndices(status)
            i = I[1]
            j = I[2]
            k = I[3]
            if status[i,j,k]
                if (!status[i+1,j,k] ||
                    !status[i-1,j,k] ||
                    !status[i,j+1,k] ||
                    !status[i,j-1,k] ||
                    !status[i,j,k+1] ||
                    !status[i,j,k-1])
                    edge[i,j,k]=true
                end
            end
        end

    	target_normal=(target_point-[.5,.5,.5])/norm(target_point-[.5,.5,.5])

        tube_eps = 2*eps
        uvals = ∞ * ones(n1, n2, n3)
        U,V,W=SphereUtils.cp_map_on_sphere_exact(n1,n2,n3,radius)
        SphereUtils.init_point_target!(uvals,
                                       status,
                                       dx,
                                       dy,
                                       dz,
                                       target_point,
                                       target_normal,
                                       tube_eps,
                                       U,
                                       V,
                                       W)

       H=(p,q,r)->sqrt(p^2+q^2+r^2)

       for i=1:niters
           SphereUtils.lax_sweep(uvals,
                                 status,
                                 edge,
                                 rhs,
                                 dx,
                                 dy,
                                 dz,
                                 sigma_x,
                                 sigma_y,
                                 sigma_z,
                                 H,
                                 U,
                                 V,
                                 W,
                                 eps)

       end

       Linf = SphereUtils.linf_error(uvals,U, V, W,target_normal)
       L1 = SphereUtils.l1_error(uvals, U, V, W,dx,dy,dz, eps,radius,target_normal)
       return L1, Linf, uvals
end

"""
This function computes the solution on the sphere when it is represented as a point
    cloud.
"""
function main_point_cloud_sphere(point_cloud::Array{Float64,2},
                                 n1::Int64,
                                 n2::Int64,
                                 n3::Int64,
                                 radius::Float64,
                                 target_point::Array{Float64,1},
                                 eps_factor::Int64,
                                 sigma_x::Float64,
                                 sigma_y::Float64,
                                 sigma_z::Float64,
                                 niters::Int64)

     dx = 1.0/(n1-1)
     dy = 1.0/(n2-1)
     dz = 1.0/(n3-1)
     eps = eps_factor*dx # For this code we assume uniform grid, but if the
     #grid is not uniform we should declare an epsilon in each direction
     U,V,W=CpMaps.compute_cp_mapping(N,N,N,point_cloud,2*eps)
     S1, S2 ,T11, T12, T13, T21, T22, T23,N1, N2, N3=DPG.CPM_info(Q,ϵ, dx, U, V, W)
     ∞ = 1000.0
     #initilize status matrix: if status[i,j,k] is true, the grid node lies
     #in the extended surface
     status=fill(false, (n1, n2,n3))
     edge=fill(false, (n1, n2,n3))
     rhs = ones(n1,n2,n3) # right hand side matrix for sphere test

     A  =∞*ones(n1,n2,n3)
     B  =∞*ones(n1,n2,n3)
     C  =∞*ones(n1,n2,n3)
     D  =∞*ones(n1,n2,n3)
  	 E  =∞*ones(n1,n2,n3)
 	 F  =∞*ones(n1,n2,n3)

 end

# L1s=ones(5)
# Linfs=ones(5)
# orders = ones(4)
sig = 1.
gridsize = [101, 201, 301, 401, 501]
for index = 5:5
    N = gridsize[index]
    L1, Linf, uvals = L1, Linf, uvals = main_exact_sphere(N,N,N,.4,[0.9, 0.5, 0.5], 4, sig,sig,sig, 200)
    L1s[index]=L1
    Linfs[index]=Linf
    if index>1
        order = log(Linfs[index]/Linfs[index-1])/log(1/gridsize[index]/(1/gridsize[index-1]))
        print(order)
        orders[index-1]=order
    end
    print(Linf)
end
#N=301
#sig = 1.
#@time L1, Linf, uvals = main_exact_sphere(N,N,N,.4,[0.9, 0.5, 0.5], 4, sig,sig,sig, 100)

#print(L1)
