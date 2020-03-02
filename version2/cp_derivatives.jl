module CpDerivs

using LinearAlgebra


function CPM_gradientcentral(U, V, W, dx)

Ux=0.0*U
Uy=0.0*U
Uz=0.0*U

Vx=0.0*V
Vy=0.0*V
Vz=0.0*V

Wx=0.0*W
Wy=0.0*W
Wz=0.0*W

N1, N2 ,N3=size(U)
#we assume uniform grid()
    for i=1:1:N1

                I_1=Neumann_idx(i-1,N1)
                I_2=Neumann_idx(i-2,N1)

                Ip1=Neumann_idx(i+1,N1)
                Ip2=Neumann_idx(i+2,N1)

        for j=1:1:N2

                J_1=Neumann_idx(j-1,N2)
                J_2=Neumann_idx(j-2,N2)

                Jp1=Neumann_idx(j+1,N2)
                Jp2=Neumann_idx(j+2,N2)

             for k=1:1:N3

                K_1=Neumann_idx(k-1,N3)
                K_2=Neumann_idx(k-2,N3);

                Kp1=Neumann_idx(k+1,N3)
                Kp2=Neumann_idx(k+2,N3)


                        Ux[i,j,k]=(-1.0*U[Ip2,j,k]+8*U[Ip1,j,k]-8*U[I_1,j,k]+U[I_2,j,k])/(12*dx)#(U[Ip1,j,k]-U[I_1,j,k])/(12*dx)
                        Vx[i,j,k]=(-1.0*V[Ip2,j,k]+8*V[Ip1,j,k]-8*V[I_1,j,k]+V[I_2,j,k])/(12*dx)#(V[Ip1,j,k]-V[I_1,j,k])/(2*dx)
                        Wx[i,j,k]=(-1.0*W[Ip2,j,k]+8*W[Ip1,j,k]-8*W[I_1,j,k]+W[I_2,j,k])/(12*dx)#(W[Ip1,j,k]-W[I_1,j,k])/(2*dx)



                        Vy[i,j,k]=(-1.0*V[i,Jp2,k]+8*V[i,Jp1,k]-8*V[i,J_1,k]+V[i,J_2,k])/(12*dx)#(V[i,Jp1,k]-V[i,J_1,k])/(2*dx)
                        Uy[i,j,k]=(-1.0*U[i,Jp2,k]+8*U[i,Jp1,k]-8*U[i,J_1,k]+U[i,J_2,k])/(12*dx)#(U[i,Jp1,k]-U[i,J_1,k])/(2*dx)
                        Wy[i,j,k]=(-1.0*W[i,Jp2,k]+8*W[i,Jp1,k]-8*W[i,J_1,k]+W[i,J_2,k])/(12*dx)#(W[i,Jp1,k]-W[i,J_1,k])/(2*dx)


                        Vz[i,j,k]=(-1.0*V[i,j,Kp2]+8*V[i,j,Kp1]-8*V[i,j,K_1]+V[i,j,K_2])/(12*dx)#(V[i,j,Kp1]-V[i,j,K_1])/(2*dx)
                        Uz[i,j,k]=(-1.0*U[i,j,Kp2]+8*U[i,j,Kp1]-8*U[i,j,K_1]+U[i,j,K_2])/(12*dx)#(U[i,j,Kp1]-U[i,j,K_1])/(2*dx)
                        Wz[i,j,k]=(-1.0*W[i,j,Kp2]+8*W[i,j,Kp1]-8*W[i,j,K_1]+W[i,j,K_2])/(12*dx)#(W[i,j,Kp1]-W[i,j,K_1])/(2*dx)


             end
         end
    end

[Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz]
end


function Neumann_idx(i, m)

        I=i;     #default value: assuming 1<= i <=m

         if i<1
                 I=1
         elseif i>m
                 I=m
         end
 return I
 end


 function cpm_jacobian_info(point_cloud::Array{Float,2},
                   ϵ,
                   U::Array{Float64,3},
                   V::Array{Float64,3},
                   W::Array{Float64,3})

     #creating Cartesian grid
     nx, ny, nz = map(dim->size(U, dim), (1,2,3))
     h=1./(nx-1)
     x_range=range(0.0, stop=1.0, length=nx)
     y_range=range(0.0, stop=1.0, length=ny)
     z_range=range(0.0, stop=1.0, length=nz)
     S1 = 1000*ones(nx,ny,nz)
     S2 = 1000*ones(nx,ny,nz)
     T11 = zeros(nx,ny,nz)
     T12 = zeros(nx,ny,nz)
     T13 = zeros(nx,ny,nz)
     T21 = zeros(nx,ny,nz)
     T22 = zeros(nx,ny,nz)
     T23 = zeros(nx,ny,nz)
     N1 = zeros(nx,ny,nz)
     N2 = zeros(nx,ny,nz)
     N3 = zeros(nx,ny,nz)

     ∞ = 1000.0

     #the central difference stencil of the closest point mapping
     DCP=zeros(3,3)
     pI=zeros(3)

     v1=zeros(3)
     v2=zeros(3)
     v3=zeros(3)

     Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz=CPM_gradientcentral(U, V, W, h)

     for I in eachindex(U)

         if U[I]>0
             i,j,k = Tuple(CartesianIndices(U)[I])

             pI[1]=x_range[i]
             pI[2]=y_range[j]
             pI[3]=z_range[k]

             dist_to_point_cloud=norm(pI-[U[i,j,k],V[i,j,k],W[i,j,k]])

             if min(i,j,k)>1 && max(i,j,k)<M && dist_to_point_cloud<=ϵ+1e-8

                 v1[1]=Ux[i,j,k]
                 v2[1]=Uy[i,j,k]
                 v3[1]=Uz[i,j,k]

                 v1[2]=Vx[i,j,k]
                 v2[2]=Vy[i,j,k]
                 v3[2]=Vz[i,j,k]

                 v1[3]=Wx[i,j,k]
                 v2[3]=Wy[i,j,k]
                 v3[3]=Wz[i,j,k]

                 #The Jacobian of the closest point mapping
                 DCP=[v1 v2 v3]

                 u,σ,v=svd(DCP)

                 S1[I]=σ[1]
                 S2[I]=σ[2]
                 T11[I]=u[1,1]
                 T12[I]=u[2,1]
                 T13[I]=u[3,1]

                 T21[I]=u[1,2]
                 T22[I]=u[2,2]
                 T23[I]=u[3,2]

                 N1[I]=u[1,3]
                 N2[I]=u[2,3]
                 N3[I]=u[3,3]


                 #end
             end
         end
     end

     return S1, S2, T11, T12, T13, T21, T22, T23, N1, N2, N3
 end
