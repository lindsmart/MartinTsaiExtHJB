#given grid node compute closest point to the surface
module CpMaps

using NearestNeighbors
using LinearAlgebra

#computes k closest points to gridpoint

function k_closests_points(kdtree, gridpoint::Array{Float64,1}, surfdata::Array{Float64,2}, k)
    #kdtree = KDTree(surfdata)
    idxs, dists = knn(kdtree, gridpoint, 1, true);


    idxs, dists = knn(kdtree, surfdata[:,idxs[1]], k, true);

return idxs
end


function init_closest_point_mapping!(point_cloud_kdtree,
                                    nearest_neighbor_list::Array{Int64,1},
                                    M::Int64,
                                    point_cloud::Array{Float64,2},
                                    ϵ::Float64,
                                    h::Float64,
                                    x_range,
                                    y_range,
                                    z_range)
    closest_points=1000*ones(M,M,M)
    n_points=size(point_cloud,2)

    linIdxs=LinearIndices(closest_points);

    idx=Array{Int64}(undef,1)

    w = Int64(ceil(ϵ/h))

    h=x_range[2]-x_range[1]

    #updating the ϵ neighborhood of each point in point_cloud
    point_ind=1
    for I=1:n_points

        q0 = point_cloud[:, I]

        i0=max(1, Int64(floor( (q0[1]-x_range[1])/ h ))-w)
        j0=max(1, Int64(floor( (q0[2]-y_range[1])/ h ))-w)
        k0=max(1, Int64(floor( (q0[3]-z_range[1])/ h ))-w)

        i1=min( length(x_range), i0+2*w)
        j1=min( length(y_range), j0+2*w)
        k1=min( length(z_range), k0+2*w)

        #need to optimize for grid ordering
        for k=k0:k1, j=j0:j1, i=i0:i1
            if closest_points[i,j,k]>=1000
                pointI = [x_range[i]; y_range[j]; z_range[k]]
                idx, dist = knn(point_cloud_kdtree, pointI, 1, true)
                closest_points[i,j,k]=dist[1]
                nearest_neighbor_list[point_ind]=linIdxs[i,j,k]
                point_ind+=1

            end
        end
    end
    return closest_points
end

#helper function to determine variables of interpolating surface
function sortvariables(index)
    if index==1
        z=1
        x=2
        y=3
        functionof="functionofyz"
    elseif index==2
        z=2
        x=1
        y=3
        functionof="functionofxz"
    else
        z=3
        x=1
        y=2
        functionof="functionofxy"
    end
    return x,y,z,functionof
end

#create "vandermonde" matrix
function formA(points::Array{Float64,2})
    A=[points[1,1]^2 points[1,1]*points[2,1] points[2,1]^2 points[1,1] points[2,1] 1.0;
       points[1,2]^2 points[1,2]*points[2,2] points[2,2]^2 points[1,2] points[2,2] 1.0;
       points[1,3]^2 points[1,3]*points[2,3] points[2,3]^2 points[1,3] points[2,3] 1.0;
       points[1,4]^2 points[1,4]*points[2,4] points[2,4]^2 points[1,4] points[2,4] 1.0;
       points[1,5]^2 points[1,5]*points[2,5] points[2,5]^2 points[1,5] points[2,5] 1.0;
       points[1,6]^2 points[1,6]*points[2,6] points[2,6]^2 points[1,6] points[2,6] 1.0]
end

#newton iterations
function newtonsiters(x0,y0,z0,F::Function,Fx::Function,Fy::Function,Fxy::Function,Fxx::Function, Fyy::Function,xinit,yinit)


   f, fx, fy, fxx, fxy, fyy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   J=zeros(2,2)

   x=[xinit, yinit]
   diff=[1.0, 1.0] #J\psi([xinit, yinit])

   psi=[0.0; 0.0]
   df=0.0

   for i=1:7

       if norm(diff)<1e-8
           break
       end

       f=F(x)
       fx=Fx(x)
       fy=Fy(x)
       fxx=Fxx(x)
       fxy=Fxy(x)
       fyy=Fyy(x)

       df=f-z0

       J[1,1]=1.0+fx^2+df*fxx
       J[1,2]=fx*fy+df*fxy
       J[2,1]=fx*fy+df*fxy
       J[2,2]=1.0+fy^2+df*fyy

       psi[1] = x[1]-x0+df*fx
       psi[2] = x[2]-y0+df*fy

       diff=J\psi
       x.-=diff

   end

   x

end

#find closest point to gridpoint on interpolating surface given surfdata
function CPonSinterp(kdtree, gridpoint::Array{Float64,1}, surfdata::Array{Float64,2},testpoints)

    #if test==1
        idxsCP=k_closests_points(kdtree, gridpoint,surfdata,testpoints)
        idxs=idxsCP[testpoints-5:end] #leave out closest point to get a better spread of 6 points
    #else
        #idxsCP=kClosestPoints(kdtree, gridpoint,surfdata,6)
        #idxs=idxsCP[:] #leave out closest point to get a better spread of 6 points
    #end
    #idxs=idxsCP[:]

    # xdiff=maximum(surfdata[1,idxs])-minimum(surfdata[1,idxs])
    # ydiff=maximum(surfdata[2,idxs])-minimum(surfdata[2,idxs])
    # zdiff=maximum(surfdata[3,idxs])-minimum(surfdata[3,idxs])
    #
    # diffvec=[xdiff,ydiff,zdiff]
    # indices=sortperm(diffvec)

#determine what variables the surface is a function of

samples1=surfdata[1,idxs]
points1=surfdata[[2,3],idxs]
functionof1="functionofyz"
A1=formA(points1)

samples2=surfdata[2,idxs]
points2=surfdata[[1,3],idxs]
functionof2="functionofxz"
A2=formA(points2)

samples3=surfdata[3,idxs]
points3=surfdata[[1,2],idxs]
functionof3="functionofxy"
A3=formA(points3)

conds = [cond(A1),cond(A2),cond(A3)]
indices=sortperm(conds)
next=false;

if indices[1] == 1 && cond(A1)<1e5
    coeffs=A1\samples1
    functionof="functionofyz"

elseif indices[1] == 2 && cond(A2)<1e5
    coeffs=A2\samples2
    functionof="functionofxz"

elseif indices[1] == 3 &&  cond(A3) < 1e5
    coeffs=A3\samples3
    functionof="functionofxy"
else
    next=true;
end


# x,y,z,functionof=sortvariables(indices[1])
# samples=surfdata[z,idxs]
# points=surfdata[[x,y],idxs]
# A=formA(points)
# next=false;
# #check if invertible
# #form all three matrices A1, A2, A3 choose the one with the smallest condition number
# #xy,xz,yz
# if cond(A)>1e5
#     #try different variables
#     x,y,z,functionof=sortvariables(indices[2])
#     samples=surfdata[z,idxs]
#     points=surfdata[[x,y],idxs]
#
#     A=formA(points)
#
#     if cond(A)>1e5
#
#     if abs(det(A))!=0
#         coeffs=A\samples
#     else
#         #try different variables
#         x,y,z,functionof=sortvariables(indices[3])
#         samples=surfdata[z,idxs]
#         points=surfdata[[x,y],idxs]
#
#         A=formA(points)
#         if abs(det(A))!=0
#             coeffs=A\samples
#         else
#             next=true;
#         end
#     end
#
# else
#     coeffs=A\samples
# end

if next
    # println("Cond A1=", cond(A1))
    # println("Cond A2=", cond(A2))
    # println("Cond A3=", cond(A3))


    if testpoints<10
        #println("here3")
        CPx,CPy,CPz=CPonSinterp(kdtree,gridpoint,surfdata, testpoints+1)

        return CPx, CPy, CPz
    else
        println("here")
        CPx= surfdata[1,idxsCP[1]]
        CPy= surfdata[2,idxsCP[1]]
        CPz= surfdata[3,idxsCP[1]]
        return CPx, CPy, CPz

    end
else

#interpolating surface definition and the derivatives
f=(x)->coeffs[1]*x[1]^2+coeffs[2]*x[1]*x[2]+coeffs[3]*x[2]^2+coeffs[4]*x[1]+coeffs[5]*x[2]+coeffs[6]
fx=(x)->2*coeffs[1]*x[1]+coeffs[2]*x[2]+coeffs[4]
fy=(x)->2*coeffs[3]*x[2]+coeffs[2]*x[1]+coeffs[5]
fxx=(x)->2*coeffs[1]
fyy=(x)->2*coeffs[3]
fxy=(x)->coeffs[2]


if functionof=="functionofxy"
    x0=gridpoint[1]
    y0=gridpoint[2]
    z0=gridpoint[3]

    xinit=surfdata[1,idxsCP[1]]
    yinit=surfdata[2,idxsCP[1]]


    x0y0=newtonsiters(x0,y0,z0,f,fx,fy,fxy,fxx,fyy,xinit,yinit)

    CPx=x0y0[1]
    CPy=x0y0[2]
    CPz=f(x0y0)

elseif functionof=="functionofxz"
    x0=gridpoint[1]
    y0=gridpoint[3]
    z0=gridpoint[2]

    xinit=surfdata[1,idxsCP[1]]
    yinit=surfdata[3,idxsCP[1]]

    x0y0=newtonsiters(x0,y0,z0,f,fx,fy,fxy,fxx,fyy,xinit,yinit)

    CPx=x0y0[1]
    CPy=f(x0y0)
    CPz=x0y0[2]


elseif functionof=="functionofyz"
    x0=gridpoint[2]
    y0=gridpoint[3]
    z0=gridpoint[1]

    xinit=surfdata[2,idxsCP[1]]
    yinit=surfdata[3,idxsCP[1]]

    x0y0=newtonsiters(x0,y0,z0,f,fx,fy,fxy,fxx,fyy,xinit,yinit)

    CPx=f(x0y0)
    CPy=x0y0[1]
    CPz=x0y0[2]

end
end
#origdist=norm(gridpoint-[surfdata[1,idxsCP[1]], surfdata[2,idxsCP[1]], surfdata[3,idxsCP[1]]])
#newdist=norm(gridpoint-[CPx, CPy ,CPz])



# t=size(idxs)
#
# currmin=1000
#start=1;
#if test==2
#    start=2;
#end

# for q=1:t[1]
#
#     tempmin=norm([CPx, CPy ,CPz]-
#             [surfdata[1,idxs[q]], surfdata[2,idxs[q]], surfdata[3,idxs[q]]])
#     if tempmin<currmin
#         currmin=tempmin
#     end
# end
#
# #firstdist=norm(gridpoint-[surfdata[1,idxsCP[1]], surfdata[2,idxsCP[1]], surfdata[3,idxsCP[1]]])
# #firstmindist=norm(gridpoint-surfdata[1,idxsCP[1]], surfdata[2,idxsCP[1]], surfdata[3,idxsCP[1]]])
#
# othermin=norm([surfdata[1,idxsCP[1]], surfdata[2,idxsCP[1]], surfdata[3,idxsCP[1]]]-
#         [surfdata[1,idxs[1]], surfdata[2,idxs[1]], surfdata[3,idxs[1]]])
#
# errordist=norm(surfdata[:,idxsCP[1]]-[CPx, CPy ,CPz])
# if abs(norm([CPx,CPy,CPz])-.5)>5.e-3
#     #println("here")
#     if testpoints<10
#         CPx,CPy,CPz= CPonSinterp(kdtree,gridpoint,surfdata, testpoints+1)
#     else
#         #println("here2")
#
#         CPx= surfdata[1,idxsCP[1]]
#         CPy= surfdata[2,idxsCP[1]]
#         CPz= surfdata[3,idxsCP[1]]
#     end
# end



return CPx,CPy,CPz
end


function compute_cp_mapping(n1::Int64,
                            n2::Int64,
                            n3::Int64,
                            point_cloud::Array{Float64, 2},
                            max_eps)

    kdtree = KDTree(point_cloud)
    dx=2/(n1-1)
    dy=2/(n2-1)
    dz=2/(n3-1)
    h=2/(n1-1)
    ∞=1000.0
    U = ∞*ones(n1,n2,n3)
    V = ∞*ones(n1,n2,n3)
    W = ∞*ones(n1,n2,n3)
    nearest_neighbor_list=zeros(Int64,n1*n2*n3)
    n_points=size(point_cloud,2)

    M=Int64(n1)

    x_range=range(-1.0, stop=1.0, length=M)
    y_range=range(-1.0, stop=1.0, length=M)
    z_range=range(-1.0, stop=1.0, length=M)
    # CP=init_closest_point_mapping!(kdtree,
    #                                nearest_neighbor_list,
    #                                M,
    #                                point_cloud,
    #                                max_eps,
    #                                h,
    #                                x_range,
    #                                y_range,
    #                                z_range)
    w = Int64(ceil(max_eps/h))
    h=x_range[2]-x_range[1]

    for I in eachindex(U)

          i,j,k = Tuple(CartesianIndices(U)[I])  #ind2sub(sizeofCP, I)

          gridpoint=[(i-1)*dx-1,(j-1)*dy-1,(k-1)*dz-1]

          #dist2Q=norm(pI-Q[:,CP[I]])
          dist2Q=norm(gridpoint)-.5#[U[i,j,k],V[i,j,k],W[i,j,k]])

          if min(i,j,k)>1 && max(i,j,k)<M && abs(dist2Q)<=max_eps+1e-6

              gridpoint=[(i-1)*dx-1,(j-1)*dy-1,(k-1)*dz-1]
              # idx, dist = knn(kdtree, gridpoint, 1, true)
              # estimated_dist=dist[1]
              # if estimated_dist<max_eps
              #     CPx,CPy,CPz = CPonSinterp(kdtree,gridpoint,point_cloud, 7)
              #     U[i,j,k] = CPx
              #     V[i,j,k] = CPy
              #     W[i,j,k] = CPz
              # end
              CPx,CPy,CPz=CPonSinterp(kdtree,gridpoint,point_cloud, 6)


             U[i,j,k]=CPx
             V[i,j,k]=CPy
             W[i,j,k]=CPz
         end
     end



return U,V,W

end

end
