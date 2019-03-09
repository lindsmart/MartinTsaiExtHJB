module DPG

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


 function CPM_info(Q, ϵ, h, U::Array{Float64,3},V::Array{Float64,3}, W::Array{Float64,3})

     sum=0.0
     weight=0.0

     N=size(Q,2)

     #creating Cartesian grid
     M=Int64(ceil(1.0/h))

     x_range=range(0.0, stop=1.0, length=M+1)
     y_range=range(0.0, stop=1.0, length=M+1)
     z_range=range(0.0, stop=1.0, length=M+1)
     S1 = 1000*ones(M+1,M+1,M+1)
     S2 = 1000*ones(M+1,M+1,M+1)
     T11 = zeros(M+1,M+1,M+1)
     T12 = zeros(M+1,M+1,M+1)
     T13 = zeros(M+1,M+1,M+1)
     T21 = zeros(M+1,M+1,M+1)
     T22 = zeros(M+1,M+1,M+1)
     T23 = zeros(M+1,M+1,M+1)
     N1 = zeros(M+1,M+1,M+1)
     N2 = zeros(M+1,M+1,M+1)
     N3 = zeros(M+1,M+1,M+1)

     ∞ = 1000.0



     #the central difference stencil of the closest point mapping
     DCP=zeros(3,3)
     Eidx, Widx, Nidx, Sidx, Fidx, Bidx=0,0,0,0,0,0

     pI=zeros(3)




     v1=zeros(3)
     v2=zeros(3)
     v3=zeros(3)

     #sizeofCP=map( dim->size(CP, dim), [1,2,3])

     J=0.0
     α=0.0
     Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz=CPM_gradientcentral(U, V, W, h)
     h=x_range[2]-x_range[1]
     four_hh=4.0*h*h
     hhh=h*h*h

     for I in eachindex(U)


         if U[I]>0
             i,j,k = Tuple(CartesianIndices(U)[I])  #ind2sub(sizeofCP, I)

             pI[1]=x_range[i]
             pI[2]=y_range[j]
             pI[3]=z_range[k]
             #dist2Q=norm(pI-Q[:,CP[I]])
             dist2Q=norm(pI-[U[i,j,k],V[i,j,k],W[i,j,k]])

             if min(i,j,k)>1 && max(i,j,k)<M && dist2Q<=ϵ+1e-8

                 #step 1: compute the central difference matrix and its singular values
                 #assumes that the central difference stencil are within bounds


                 #Eidx=CP[i+1,j,k]
                 #Widx=CP[i-1,j,k]
                 #Nidx=CP[i,j+1,k]
                 #Sidx=CP[i,j-1,k]
                 #Fidx=CP[i,j,k+1]
                 #Bidx=CP[i,j,k-1]

                 #if Eidx!=-1 && Widx!=-1 && Nidx!=-1 && Sidx!=-1 && Fidx!=-1 && Bidx!=-1

                     #This matrix should be replaced by the more accurate third order finite differencing routine
                     v1[1]=Ux[i,j,k]
                     v2[1]=Uy[i,j,k]
                     v3[1]=Uz[i,j,k]

                     v1[2]=Vx[i,j,k]
                     v2[2]=Vy[i,j,k]
                     v3[2]=Vz[i,j,k]

                     v1[3]=Wx[i,j,k]
                     v2[3]=Wy[i,j,k]
                     v3[3]=Wz[i,j,k]

                     #=v1[1]=(U[i+1,j,k]-U[i-1,j,k])/h/2.0
                     v1[2]=(V[i+1,j,k]-V[i-1,j,k])/h/2.0
                     v1[3]=(W[i+1,j,k]-W[i-1,j,k])/h/2.0


                     v2[1]=(U[i,j+1,k]-U[i,j-1,k])/h/2.0
                     v2[2]=(V[i,j+1,k]-V[i,j-1,k])/h/2.0
                     v2[3]=(W[i,j+1,k]-W[i,j-1,k])/h/2.0


                     v3[1]=(U[i,j,k+1]-U[i,j,k-1])/h/2.0
                     v3[2]=(V[i,j,k+1]-V[i,j,k-1])/h/2.0
                     v3[3]=(W[i,j,k+1]-W[i,j,k-1])/h/2.0=#

                     #v1=(Q[:,Eidx]-Q[:, Widx])/h/2.0
                     #v2=(Q[:,Nidx]-Q[:, Sidx])/h/2.0
                     #v3=(Q[:,Fidx]-Q[:, Bidx])/h/2.0
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




end


#=function CPM_gradient(U, V, W, dx)

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

                # try two different third order one-sided approximations
                    use_clean_one_sided=false

                    if (use_clean_one_sided)
                        Uxp= (-3.0*U[i,j,k]+4.0*U[Ip1,j,k]-U[Ip2,j,k])/2.0/dx
                        Uxn= (3.0*U[i,j,k]-4.0*U[I_1,j,k]+U[I_2,j,k])/2.0/dx

                        Vxp= (-3.0*V[i,j,k]+4.0*V[Ip1,j,k]-V[Ip2,j,k])/2.0/dx
                        Vxn= ( 3.0*V[i,j,k]-4.0*V[I_1,j,k]+V[I_2,j,k])/2.0/dx

                        Wxp= (-3.0*W[i,j,k]+4.0*W[Ip1,j,k]-W[Ip2,j,k])/2.0/dx
                        Wxn= ( 3.0*W[i,j,k]-4.0*W[I_1,j,k]+W[I_2,j,k])/2.0/dx
                    else
                        Uxp=-(3.0*U[i,j,k]+2.0*U[I_1,j,k]-6.0*U[Ip1,j,k]+1.0*U[Ip2,j,k])/6.0/dx
                        Uxn= (3.0*U[i,j,k]+2.0*U[Ip1,j,k]-6.0*U[I_1,j,k]+1.0*U[I_2,j,k])/6.0/dx

                        Vxp=-(3.0*V[i,j,k]+2.0*V[I_1,j,k]-6.0*V[Ip1,j,k]+1.0*V[Ip2,j,k])/6.0/dx
                        Vxn= (3.0*V[i,j,k]+2.0*V[Ip1,j,k]-6.0*V[I_1,j,k]+1.0*V[I_2,j,k])/6.0/dx

                        Wxp=-(3.0*W[i,j,k]+2.0*W[I_1,j,k]-6.0*W[Ip1,j,k]+1.0*W[Ip2,j,k])/6.0/dx
                        Wxn= (3.0*W[i,j,k]+2.0*W[Ip1,j,k]-6.0*W[I_1,j,k]+1.0*W[I_2,j,k])/6.0/dx
                    end
                    #use D+D- at [i-11,j,k] and [i+1,j,k] as smoothness
                    #indicator
                    use_Ux=true

                    if use_Ux
                        Sn=U[i,j,k]-2.0*U[I_1,j,k]+U[I_2,j,k]
                        Sp=U[i,j,k]-2.0*U[Ip1,j,k]+U[Ip2,j,k]
                    else
                        Sn=V[i,j,k]-2.0*V[I_1,j,k]+V[I_2,j,k]
                        Sp=V[i,j,k]-2.0*V[Ip1,j,k]+V[Ip2,j,k]
                    end

                    if abs(Sp)<=abs(Sn)
                        Ux[i,j,k]=Uxp
                        Vx[i,j,k]=Vxp
                        Wx[i,j,k]=Wxp
                    else
                        Ux[i,j,k]=Uxn
                        Vx[i,j,k]=Vxn
                        Wx[i,j,k]=Wxn
                    end

                    Uyp=-(3.0*U[i,j,k]+2.0*U[i,J_1,k]-6.0*U[i,Jp1,k]+1.0*U[i,Jp2,k])/6.0/dx
                    Uyn= (3.0*U[i,j,k]+2.0*U[i,Jp1,k]-6.0*U[i,J_1,k]+1.0*U[i,J_2,k])/6.0/dx
                    #Uyp= (-3.0*U[i,j,k]+4.0*U[i,Jp1,k]-U[i,Jp2,k])/2.0/dx
                    #Uyn= ( 3.0*U[i,j,k]-4.0*U[i,J_1,k]+U[i,J_2,k])/2.0/dx

                    Vyp=-(3.0*V[i,j,k]+2.0*V[i,J_1,k]-6.0*V[i,Jp1,k]+1.0*V[i,Jp2,k])/6.0/dx
                    Vyn= (3.0*V[i,j,k]+2.0*V[i,Jp1,k]-6.0*V[i,J_1,k]+1.0*V[i,J_2,k])/6.0/dx
                    #Vyp= (-3.0*V[i,j,k]+4.0*V[i,Jp1,k]-V[i,Jp2,k])/2.0/dx
                    #Vyn= ( 3.0*V[i,j,k]-4.0*V[i,J_1,k]+V[i,J_2,k])/2.0/dx

                    Wyp=-(3.0*W[i,j,k]+2.0*W[i,J_1,k]-6.0*W[i,Jp1,k]+1.0*W[i,Jp2,k])/6.0/dx
                    Wyn= (3.0*W[i,j,k]+2.0*W[i,Jp1,k]-6.0*W[i,J_1,k]+1.0*W[i,J_2,k])/6.0/dx

                    if use_Ux
                        Sn=U[i,j,k]-2.0*U[i,J_1,k]+U[i,J_2,k]
                        Sp=U[i,j,k]-2.0*U[i,Jp1,k]+U[i,Jp2,k]
                    else
                        Sn=V[i,j,k]-2.0*V[i,J_1,k]+V[i,J_2,k];
                        Sp=V[i,j,k]-2.0*V[i,Jp1,k]+V[i,Jp2,k]
                    end

                    if abs(Sp)<=abs(Sn)
                        Vy[i,j,k]=Vyp
                        Uy[i,j,k]=Uyp
                        Wy[i,j,k]=Wyp
                    else
                        Vy[i,j,k]=Vyn
                        Uy[i,j,k]=Uyn
                        Wy[i,j,k]=Wyn
                    end

                    Uzp=-(3.0*U[i,j,k]+2.0*U[i,j,K_1]-6.0*U[i,j,Kp1]+1.0*U[i,j,Kp2])/6.0/dx
                    Uzn= (3.0*U[i,j,k]+2.0*U[i,j,Kp1]-6.0*U[i,j,K_1]+1.0*U[i,j,K_2])/6.0/dx
                    #Uyp= (-3.0*U[i,j,k]+4.0*U[i,j,Kp1]-U[i,j,Kp2])/2.0/dx
                    #Uyn= ( 3.0*U[i,j,k]-4.0*U[i,j,K_1]+U[i,j,K_2])/2.0/dx

                    Vzp=-(3.0*V[i,j,k]+2.0*V[i,j,K_1]-6.0*V[i,j,Kp1]+1.0*V[i,j,Kp2])/6.0/dx
                    Vzn= (3.0*V[i,j,k]+2.0*V[i,j,Kp1]-6.0*V[i,j,K_1]+1.0*V[i,j,K_2])/6.0/dx
                    #Vyp= (-3.0*V[i,j,k]+4.0*V[i,j,Kp1]-V[i,j,Kp2])/2.0/dx
                    #Vyn= ( 3.0*V[i,j,k]-4.0*V[i,j,K_1]+V[i,j,K_2])/2.0/dx

                    Wzp=-(3.0*W[i,j,k]+2.0*W[i,j,K_1]-6.0*W[i,j,Kp1]+1.0*W[i,j,Kp2])/6.0/dx
                    Wzn= (3.0*W[i,j,k]+2.0*W[i,j,Kp1]-6.0*W[i,j,K_1]+1.0*W[i,j,K_2])/6.0/dx

                    if use_Ux
                        Sn=U[i,j,k]-2.0*U[i,j,K_1]+U[i,j,K_2]
                        Sp=U[i,j,k]-2.0*U[i,j,Kp1]+U[i,j,Kp2]
                    else
                        Sn=V[i,j,k]-2.0*V[i,j,K_1]+V[i,j,K_2]
                        Sp=V[i,j,k]-2.0*V[i,j,Kp1]+V[i,j,Kp2]
                    end

                    if abs(Sp)<=abs(Sn)
                        Vz[i,j,k]=Vzp
                        Uz[i,j,k]=Uzp
                        Wz[i,j,k]=Wzp
                    else
                        Wz[i,j,k]=Vzn
                        Uz[i,j,k]=Uzn
                        Wz[i,j,k]=Wzn
                    end

             end
         end
    end

[Ux, Uy, Uz, Vx, Vy, Vz, Wx, Wy, Wz]
end=#
