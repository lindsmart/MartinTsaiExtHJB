using MATLAB
using NearestNeighbors
using LinearAlgebra
if true
    include("./CPmap.jl")
    include("./DPG.jl")
    include("./LaxFriedFS.jl")
    include("./LaxFriedDPGexact.jl")
    include("./LaxFriedMemSap.jl")
    include("./ErrorComp.jl")
end


function mainTorus(N,nitrs,Q::Array{Float64},nT,T);


	dx=1/(N-1)
    U,V,W=CPmap.compUVW(N,N,N,Q,8*dx)
    ϵ=5*dx
    S1, S2 ,T11, T12, T13, T21, T22, T23,N1, N2, N3=DPG.CPM_info(Q,ϵ, dx, U, V, W)


	dx=dy=dz=1/(N-1)

    n1=n2=n3=N




	∞ = 1000.0



	Uvals  =fill(∞, (n1, n2, n3))

	A  =ones(n1,n2,n3)
	B  =ones(n1,n2,n3)
	C  =ones(n1,n2,n3)
	D  =ones(n1,n2,n3)
	E  =ones(n1,n2,n3)
	F  =ones(n1,n2,n3)
	for i in 1:1:n1
		for j in 1:1:n2
			for k in 1:1:n3
				if S1[i,j,k]<1000
					A[i,j,k]=T11[i,j,k]^2/S1[i,j,k]^2+T21[i,j,k]^2/S2[i,j,k]^2+ N1[i,j,k]^2
					B[i,j,k]=T12[i,j,k]^2/S1[i,j,k]^2+T22[i,j,k]^2/S2[i,j,k]^2+ N2[i,j,k]^2
					C[i,j,k]=T13[i,j,k]^2/S1[i,j,k]^2+T23[i,j,k]^2/S2[i,j,k]^2+ N3[i,j,k]^2
					D[i,j,k]=(T11[i,j,k]*T12[i,j,k])/S1[i,j,k]^2+(T21[i,j,k]*T22[i,j,k])/S2[i,j,k]^2 + N1[i,j,k]*N2[i,j,k]
					E[i,j,k]=(T12[i,j,k]*T13[i,j,k])/S1[i,j,k]^2+(T22[i,j,k]*T23[i,j,k])/S2[i,j,k]^2 + N2[i,j,k]*N3[i,j,k]
					F[i,j,k]=(T11[i,j,k]*T13[i,j,k])/S1[i,j,k]^2+(T21[i,j,k]*T23[i,j,k])/S2[i,j,k]^2 + N1[i,j,k]*N3[i,j,k]


				end
			end
		end
	end



	status=fill(true, (n1, n2,n3))
	edge=fill(false, (n1, n2,n3))
	initial=fill(false, (n1, n2,n3))
	X = ones(n1,n2,n3)
	#u=.4;
	#l=.2;
	for i in 1:1:n1
		for j in 1:1:n2
			for k in 1:1:n3
				if S1[i,j,k]==1000
					X[i,j,k]=∞
					status[i,j,k]=false
					initial[i,j,k]=true

				end
			end
		end
	end

	for i=1:1:n1
	    for j=1:1:n2
			for k=1:1:n3
			if S1[i,j,k]<1000
				if (S1[i+1,j,k]==1000 || S1[i-1,j,k]==1000 || S1[i,j+1,k]==1000 || S1[i,j-1,k]==1000||
					S1[i,j,k+1]==1000 || S1[i,j,k-1]==1000)
					edge[i,j,k]=true;
				end
			end
			end
	    end
	end



	epsi=.05
	LxFSweep.init_point_target!(Uvals, initial,X, dx,dy,dz, nT, T,epsi)

	#hamiltonian for eikonal case
	#maxsig1sq=maximum(1 ./(S1))
	#maxsig2sq=maximum(1 ./(S2))
	#maxeig=max(maxsig1sq,maxsig2sq)
	#maxeig=max(maxsig1sq,1) #or mu
	σx=0;#maxeig #sqrt(maxsig1sq+maxsig2sq+1)
	σy=0;#maxeig #sqrt(maxsig1sq+maxsig2sq+1)
	σz=0;#maxeig #sqrt(maxsig1sq+maxsig2sq+1)
	H=(i,j,k,p,q,r)->sqrt(A[i,j,k]*p^2+B[i,j,k]*q^2+C[i,j,k]*r^2+D[i,j,k]*p*q+E[i,j,k]*q*r+F[i,j,k]*p*r)

	##hamiltonion for eikonal case euclidean space
	#σx=1
	#σy=1
	#σz=1
	#H=(i,j,k,p,q,r)->sqrt(p^2+q^2+r^2)


	for i=1:nitrs

		#println("sweep iteration ", i)

		LxFSweep.laxsweep(Uvals, status, edge, initial, X,  dx,dy,dz,σx,σy,σz, H, S1,S2, U,V,W,ϵ)

	end



	#contour(U[:,51,:],[i for i=0:4/100:4])
	#axis("square")
	return Uvals,U,V,W
end

mf= MatFile("PtCloudtorusv2.mat");
Q=get_variable(mf, "CPun");

close(mf)

R=.2;r=.1;u=pi/2;v=pi/4;
T=[(R+r.*cos(v)).*cos(u)+.5,(R+r.*cos(v)).*sin(u)+.5,r.*sin(v)+.5]
t1=[-sin(v),cos(v),0]
t2=[-sin(v)*cos(u),-sin(v)*sin(u) ,cos(v)]
nT=cross(t1,t2)

@time Uvals,U,V,W=mainTorus(201,100,Q,nT,T);
