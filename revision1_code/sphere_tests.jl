module SphereTests
using DocStringExtensions
using LinearAlgebra
include("./sphere_utils.jl")
include("./closest_point_mappings.jl")
include("./cp_derivatives.jl")
using .SphereUtils
using .CpMaps
using .DPG
"""
These functions compute the distance function on the sphere to a target point.
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
        - n_iters: number of sweeping iterations (one iteration includes 8 sweeping directions)
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


        dx = 2.0/(n1-1)
        dy = 2.0/(n2-1)
        dz = 2.0/(n3-1)
        eps =eps_factor*dx  #For this code we assume uniform grid, but if the
        #grid is not uniform we should declare an epsilon in each direction

        ∞ = 1000.0
        #initilize status matrix: if status[i,j,k] is true, the grid node lies
        #in the extended surface
        status=fill(false, (n1, n2,n3))
        edge=fill(false, (n1, n2,n3))
        rhs = ∞ * ones(n1,n2,n3) # right hand side matrix for sphere test
        num_in_eps = 0
        for I in CartesianIndices(status)
            i = I[1]
            j = I[2]
            k = I[3]
            x = (i-1)*dx-1
            y = (j-1)*dy-1
            z = (k-1)*dz-1
            r = sqrt((x-0.0)^2 +(y-0.0)^2+(z-0.0)^2)
            if abs(r - radius) < eps + 1e-13
                status[i,j,k] = true
                num_in_eps = num_in_eps + 1
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


    	target_normal=(target_point-[0.0,0.0,0.0])/norm(target_point-[0.0,0.0,0.0])


        uvals = ∞ * ones(n1, n2, n3)
        U,V,W=SphereUtils.cp_map_on_sphere_exact(n1,n2,n3,radius)
        tube_eps = 1.25*eps
        # SphereUtils.init_point_target!(uvals,
        #                                status,
        #                                dx,
        #                                dy,
        #                                dz,
        #                                target_point,
        #                                target_normal,
        #                                tube_eps,
        #                                U,
        #                                V,
        #                                W)
        SphereUtils.init_point_target_high_order_v2!(uvals,
                                                  status,
                                                  dx,
                                                  dy,
                                                  dz,
                                                  target_point,
                                                  target_normal,
                                                  U,
                                                  V,
                                                  W)

       H=(i,j,k,p,q,r)->sqrt(p^2+q^2+r^2)
       num_iters = 0

       for i=1:niters
           num_iters +=1
           time = @elapsed error = SphereUtils.lax_sweep(uvals,
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
            if error < 1.e-13
                break
            end
        end

        Linf, ijk = SphereUtils.linf_error(uvals,U, V, W,edge,status,radius,target_normal)
        L1, num_pts  = SphereUtils.l1_error(uvals,U, V, W, edge, status, dx,dy,dz, eps,radius,target_normal)

        return L1, Linf, num_iters, num_in_eps#, status, ijk

end

"""
This function computes the solution on the sphere when it is represented exactly
    using the 3rd order Lax Friedrichs method.
"""
function main_exact_sphere_high_order(n1::Int64,
                                      n2::Int64,
                                      n3::Int64,
                                      radius::Float64,
                                      target_point::Array{Float64,1},
                                      eps_factor::Int64,
                                      sigma_x::Float64,
                                      sigma_y::Float64,
                                      sigma_z::Float64,
                                      niters::Int64)

    #Initalize by solving the first order problem first.
    dx = 2.0/(n1-1)
    dy = 2.0/(n2-1)
    dz = 2.0/(n3-1)
    eps = eps_factor*dx #For this code we assume uniform grid, but if the
    #grid is not uniform we should declare an epsilon in each direction

    U,V,W=SphereUtils.cp_map_on_sphere_exact(n1,n2,n3,radius)
    status=fill(false, (n1, n2,n3))
    edge=fill(false, (n1, n2,n3))
    ∞ = 1000.0
    rhs = ∞ * ones(n1,n2,n3) # right hand side matrix for sphere test
    num_in_eps = 0
    for I in CartesianIndices(status)
        i = I[1]
        j = I[2]
        k = I[3]
        x = (i-1)*dx-1
        y = (j-1)*dy-1
        z = (k-1)*dz-1
        r = sqrt(x^2 +y^2+z^2)
        if abs(r - radius) < eps + 1e-13
            status[i,j,k] = true
            num_in_eps = num_in_eps + 1
            rhs[i,j,k] = 1 - (r - radius)/r
        end


    end

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
                !status[i,j,k-1] ||
                !status[i+2,j,k] ||
                !status[i-2,j,k] ||
                !status[i,j+2,k] ||
                !status[i,j-2,k] ||
                !status[i,j,k+2] ||
                !status[i,j,k-2])
                edge[i,j,k]=true
            end
        end
    end

    target_normal=target_point/radius

    tube_eps = 1.25*eps
    uvals = ∞ * ones(n1, n2, n3)


    SphereUtils.init_point_target_high_order_v2!(uvals,
                                              status,
                                              dx,
                                              dy,
                                              dz,
                                              target_point,
                                              target_normal,
                                              U,
                                              V,
                                              W)
    H=(i,j,k,p,q,r)->sqrt(p^2+q^2+r^2)
    num_iters = 0

    for i=1:niters
        num_iters +=1
        error = SphereUtils.lax_sweep_high_order(uvals,
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
        if error < 1.e-13
            break
        end
   end
   Linf, ijk= SphereUtils.linf_error(uvals,U, V, W,edge,status,radius,target_normal)

   L1, num_pts, avgError  = SphereUtils.l1_error(uvals,U, V, W, edge, status, dx,dy,dz, eps,radius,target_normal)

   return L1, Linf, avgError, num_iters, num_in_eps#, status,ijk

end

"""
This function computes the solution on the sphere when it is represented by a
    point cloud using the 1st order Lax Friedrichs method.
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

     dx = 2.0/(n1-1)
     dy = 2.0/(n2-1)
     dz = 2.0/(n3-1)
     eps =eps_factor*dx # #For this code we assume uniform grid, but if the
     #grid is not uniform we should declare an epsilon in each direction
     U,V,W = CpMaps.compute_cp_mapping(n1,n2,n3,point_cloud,4*eps)


     S1, S2 ,T11, T12, T13, T21, T22, T23,N1, N2, N3 = DPG.CPM_info(eps,
                                                                    dx,
                                                                    U,
                                                                    V,
                                                                    W)

     ∞ = 1000.0
     #initilize status matrix: if status[i,j,k] is true, the grid node lies
     #in the extended surface
     status=fill(false, (n1, n2,n3))
     edge=fill(false, (n1, n2,n3))
     rhs = ones(n1,n2,n3) # right hand side matrix for sphere test

     A = ∞*ones(n1,n2,n3)
     B = ∞*ones(n1,n2,n3)
     C = ∞*ones(n1,n2,n3)
     D = ∞*ones(n1,n2,n3)
  	 E = ∞*ones(n1,n2,n3)
 	 F = ∞*ones(n1,n2,n3)
     A11 = ∞*ones(n1,n2,n3)
     A12 = ∞*ones(n1,n2,n3)
     A13 = ∞*ones(n1,n2,n3)
     A22 = ∞*ones(n1,n2,n3)
     A23 = ∞*ones(n1,n2,n3)
     A33 = ∞*ones(n1,n2,n3)
     num_in_eps = 0
     for I in CartesianIndices(status)
         i = I[1]
         j = I[2]
         k = I[3]
         x = (i-1)*dx -1
         y = (j-1)*dy -1
         z = (k-1)*dz -1
         r = sqrt((x)^2 +(y)^2+(z)^2)
         if abs(r - radius) < eps + 1e-13
             status[i,j,k] = true
             num_in_eps = num_in_eps + 1
             A[i,j,k] = T11[i,j,k]^2/S1[i,j,k]^2+T21[i,j,k]^2/S2[i,j,k]^2+ N1[i,j,k]^2
             B[i,j,k] = T12[i,j,k]^2/S1[i,j,k]^2+T22[i,j,k]^2/S2[i,j,k]^2+ N2[i,j,k]^2
             C[i,j,k] = T13[i,j,k]^2/S1[i,j,k]^2+T23[i,j,k]^2/S2[i,j,k]^2+ N3[i,j,k]^2
             D[i,j,k] = (T11[i,j,k]*T12[i,j,k])/S1[i,j,k]^2+(T21[i,j,k]*T22[i,j,k])/S2[i,j,k]^2 + N1[i,j,k]*N2[i,j,k]
             E[i,j,k] = (T12[i,j,k]*T13[i,j,k])/S1[i,j,k]^2+(T22[i,j,k]*T23[i,j,k])/S2[i,j,k]^2 + N2[i,j,k]*N3[i,j,k]
             F[i,j,k] = (T11[i,j,k]*T13[i,j,k])/S1[i,j,k]^2+(T21[i,j,k]*T23[i,j,k])/S2[i,j,k]^2 + N1[i,j,k]*N3[i,j,k]
             A11[i,j,k] = T11[i,j,k]^2/S1[i,j,k] + T21[i,j,k]^2/S2[i,j,k] + N1[i,j,k]^2
             A12[i,j,k] = T11[i,j,k]*T12[i,j,k]/S1[i,j,k] + T21[i,j,k]*T22[i,j,k]/S2[i,j,k] + N1[i,j,k]*N2[i,j,k]
             A13[i,j,k] = T11[i,j,k]*T13[i,j,k]/S1[i,j,k]+T21[i,j,k]*T23[i,j,k]/S2[i,j,k]+ N1[i,j,k]*N3[i,j,k]
             A22[i,j,k] = T12[i,j,k]^2/S1[i,j,k]+T22[i,j,k]^2/S2[i,j,k] + N2[i,j,k]^2
             A23[i,j,k] = T12[i,j,k]*T13[i,j,k]/S1[i,j,k]+T22[i,j,k]*T23[i,j,k]/S2[i,j,k]+ N2[i,j,k]*N3[i,j,k]
             A33[i,j,k] = T13[i,j,k]^2/S1[i,j,k]+T23[i,j,k]^2/S2[i,j,k] + N3[i,j,k]^2
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
                 edge[i,j,k] = true
             end
         end
     end

     target_normal=(target_point)/radius

     uvals = ∞ * ones(n1, n2, n3)
     SphereUtils.init_point_target_high_order_v2!(uvals,
                                                 status,
                                                 dx,
                                                 dy,
                                                 dz,
                                                 target_point,
                                                 target_normal,
                                                 U,
                                                 V,
                                                 W)
     # tube_eps = 1.25*eps
     # SphereUtils.init_point_target!(uvals,
     #                                status,
     #                                dx,
     #                                dy,
     #                                dz,
     #                                target_point,
     #                                target_normal,
     #                                tube_eps,
     #                                U,
     #                                V,
     #                                W)

     H=(i,j,k,p,q,r)->norm([A11[i,j,k] A12[i,j,k] A13[i,j,k];
                            A12[i,j,k] A22[i,j,k] A23[i,j,k];
                            A13[i,j,k] A23[i,j,k] A33[i,j,k]]*[p,q,r])#
    #H=(i,j,k,p,q,r)->sqrt(A[i,j,k]*p^2+B[i,j,k]*q^2+C[i,j,k]*r^2+D[i,j,k]*p*q+E[i,j,k]*q*r+F[i,j,k]*p*r)
    num_iters = 0
    # sigma_x = .5*(2*maximum(A[A.<1000]) + maximum(D[D.<1000])+maximum(F[F.<1000]))
    # sigma_y = .5*(2*maximum(B[B.<1000]) + maximum(D[D.<1000])+maximum(E[E.<1000]))
    # sigma_z = .5*(2*maximum(C[C.<1000]) + maximum(E[E.<1000])+maximum(F[F.<1000]))
    for i=1:niters
        num_iters +=1
        error = SphereUtils.lax_sweep_point_cloud(uvals,
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
                                                 S1,
                                                 S2,
                                                 eps)
        if error < 1.e-13
            break
        end
   end
   Linf, ijk= SphereUtils.linf_error(uvals,U, V, W,edge,status,radius,target_normal)
   L1, num_pts, avgError  = SphereUtils.l1_error(uvals,U, V, W, edge, status, dx,dy,dz, eps,radius,target_normal)

   return L1, Linf, avgError, num_iters, num_in_eps
 end

 """
 This function computes the solution on the sphere when it is represented by a
     point cloud using the 3rd order Lax Friedrichs method.
 """

function main_point_cloud_sphere_high_order(point_cloud::Array{Float64,2},
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

     dx = 2.0/(n1-1)
     dy = 2.0/(n2-1)
     dz = 2.0/(n3-1)
     ϵ =eps_factor*dx #For this code we assume uniform grid, but if the
     #grid is not uniform we should declare an epsilon in each direction
     U,V,W = CpMaps.compute_cp_mapping(n1,n2,n3,point_cloud,2*ϵ)

     S1, S2 ,T11, T12, T13, T21, T22, T23, N1, N2, N3 = DPG.CPM_info(ϵ,
                                                                    dx,
                                                                    U,
                                                                    V,
                                                                    W)

     ∞ = 1000.0
     #initilize status matrix: if status[i,j,k] is true, the grid node lies
     #in the extended surface
     status=fill(false, (n1, n2,n3))
     edge=fill(false, (n1, n2,n3))
     rhs = ones(n1,n2,n3) # right hand side matrix for sphere test

     A = ∞*ones(n1,n2,n3)
     B = ∞*ones(n1,n2,n3)
     C = ∞*ones(n1,n2,n3)
     D = ∞*ones(n1,n2,n3)
  	 E = ∞*ones(n1,n2,n3)
 	 F = ∞*ones(n1,n2,n3)
     A11 = ∞*ones(n1,n2,n3)
     A12 = ∞*ones(n1,n2,n3)
     A13 = ∞*ones(n1,n2,n3)
     A22 = ∞*ones(n1,n2,n3)
     A23 = ∞*ones(n1,n2,n3)
     A33 = ∞*ones(n1,n2,n3)

     num_in_eps = 0
     for I in CartesianIndices(status)
         i = I[1]
         j = I[2]
         k = I[3]
         x = (i-1)*dx -1
         y = (j-1)*dy -1
         z = (k-1)*dz -1
         r = sqrt((x)^2 +(y)^2+(z)^2)
         if abs(r - radius) < ϵ + 1e-13
             status[i,j,k] = true
             num_in_eps = num_in_eps + 1
             A[i,j,k] = T11[i,j,k]^2/S1[i,j,k]^2+T21[i,j,k]^2/S2[i,j,k]^2+ N1[i,j,k]^2
             B[i,j,k] = T12[i,j,k]^2/S1[i,j,k]^2+T22[i,j,k]^2/S2[i,j,k]^2+ N2[i,j,k]^2
             C[i,j,k] = T13[i,j,k]^2/S1[i,j,k]^2+T23[i,j,k]^2/S2[i,j,k]^2+ N3[i,j,k]^2
             D[i,j,k] = (T11[i,j,k]*T12[i,j,k])/S1[i,j,k]^2+(T21[i,j,k]*T22[i,j,k])/S2[i,j,k]^2 + N1[i,j,k]*N2[i,j,k]
             E[i,j,k] = (T12[i,j,k]*T13[i,j,k])/S1[i,j,k]^2+(T22[i,j,k]*T23[i,j,k])/S2[i,j,k]^2 + N2[i,j,k]*N3[i,j,k]
             F[i,j,k] = (T11[i,j,k]*T13[i,j,k])/S1[i,j,k]^2+(T21[i,j,k]*T23[i,j,k])/S2[i,j,k]^2 + N1[i,j,k]*N3[i,j,k]
             A11[i,j,k] = T11[i,j,k]^2/S1[i,j,k] + T21[i,j,k]^2/S2[i,j,k] + N1[i,j,k]^2
             A12[i,j,k] = T11[i,j,k]*T12[i,j,k]/S1[i,j,k] + T21[i,j,k]*T22[i,j,k]/S2[i,j,k] + N1[i,j,k]*N2[i,j,k]
             A13[i,j,k] = T11[i,j,k]*T13[i,j,k]/S1[i,j,k]+T21[i,j,k]*T23[i,j,k]/S2[i,j,k]+ N1[i,j,k]*N3[i,j,k]
             A22[i,j,k] = T12[i,j,k]^2/S1[i,j,k]+T22[i,j,k]^2/S2[i,j,k] + N2[i,j,k]^2
             A23[i,j,k] = T12[i,j,k]*T13[i,j,k]/S1[i,j,k]+T22[i,j,k]*T23[i,j,k]/S2[i,j,k]+ N2[i,j,k]*N3[i,j,k]
             A33[i,j,k] = T13[i,j,k]^2/S1[i,j,k]+T23[i,j,k]^2/S2[i,j,k] + N3[i,j,k]^2




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
                 !status[i,j,k-1] ||
                 !status[i+2,j,k] ||
                 !status[i-2,j,k] ||
                 !status[i,j+2,k] ||
                 !status[i,j-2,k] ||
                 !status[i,j,k+2] ||
                 !status[i,j,k-2])
                 edge[i,j,k]=true
             end
         end
     end

     target_normal=(target_point)/radius

     uvals = ∞ * ones(n1, n2, n3)
     SphereUtils.init_point_target_high_order_v2!(uvals,
                                                 status,
                                                 dx,
                                                 dy,
                                                 dz,
                                                 target_point,
                                                 target_normal,
                                                 U,
                                                 V,
                                                 W)

    H=(i,j,k,p,q,r)->norm([A11[i,j,k] A12[i,j,k] A13[i,j,k];
                           A12[i,j,k] A22[i,j,k] A23[i,j,k];
                           A13[i,j,k] A23[i,j,k] A33[i,j,k]]*[p,q,r])
    num_iters = 0

    sigma_x = .5*(2*maximum(A[A.<1000]) + maximum(D[D.<1000])+maximum(F[F.<1000]))#/sqrt(minimum(A[A.<1000]))
    sigma_y = .5*(2*maximum(B[B.<1000]) + maximum(D[D.<1000])+maximum(E[E.<1000]))#/sqrt(minimum(B[B.<1000]))
    sigma_z = .5*(2*maximum(C[C.<1000]) + maximum(E[E.<1000])+maximum(F[F.<1000]))#/sqrt(minimum(C[C.<1000]))
    for i=1:niters
        num_iters +=1
        error = SphereUtils.lax_sweep_point_cloud_high_order(uvals,
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
                                                             S1,
                                                             S2,
                                                             ϵ)
        if error < 1.e-13
            break
        end
   end
   Linf, ijk= SphereUtils.linf_error(uvals,U, V, W,edge,status,radius,target_normal)

   L1, num_pts, avgError  = SphereUtils.l1_error(uvals,U, V, W, edge, status, dx,dy,dz, ϵ,radius,target_normal)
   return L1, Linf, avgError, num_iters, num_in_eps,uvals
 end


end
