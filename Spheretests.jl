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

function init_point_target!(u::Array{Float64}, s::Array{Bool},X::Array{Float64}, dx,dy,dz, nT, T,epsi,
	U::Array{Float64},V::Array{Float64},W::Array{Float64})
	N=size(u,1)
	Ttilde=zeros(3,1001)
	dT=2*epsi/1000
	for i=0:1000
			alpha=i/1000
			Ttilde[:,i+1]=alpha*(T+epsi*nT)+(1-alpha)*(T-epsi*nT)
	end
	Ttree = KDTree(Ttilde)
	for a=0:1000

		x=Ttilde[1,a+1]
		y=Ttilde[2,a+1]
		z=Ttilde[3,a+1]
		iminus=(floor(Int,(x)/dx))+1;


		jminus=(floor(Int,(y)/dy))+1;


		kminus=(floor(Int,(z)/dz))+1;


		A_i = (iminus-1):(iminus+2)
		A_j = (jminus-1):(jminus+2)
		A_k = (kminus-1):(kminus+2)





		NBRS = [[i,j,k] for i in A_i, j in A_j, k in A_k]

		for n=1:64
			currPt=NBRS[n]
			i=currPt[1]
			j=currPt[2]
			k=currPt[3]
			if s[i,j,k]==false && X[i,j,k]<1000
				xyz=[(i-1)*dx ,(j-1)*dy ,(k-1)*dz]
				#ind,d=LxFSweep.kClosestPointsandDists(Ttree,xyz,Ttilde,1)

				u[i,j,k]=ErrorComp.exactdist(U[i,j,k],V[i,j,k],W[i,j,k],nT)#norm(Ttilde[:,ind[1]]-xyz)/X[i,j,k]
				s[i,j,k]=true
			end
		end

	end
end

function mainPGest(N,nitrs,Q) #method with using point cloud for sphere

	dx=1/(N-1)
	radius=.4
	ϵ=4*dx
    U,V,W=CPmap.compUVW(N,N,N,Q,8*dx)

    S1, S2 ,T11, T12, T13, T21, T22, T23,N1, N2, N3=DPG.CPM_info(Q,ϵ, dx, U, V, W)

    dx=dy=dz=1/(N-1)

    n1=n2=n3=N

    ∞ = 1000.0


    Uvals  =fill(∞, (n1, n2, n3))
	A  =∞*ones(n1,n2,n3)
	B  =∞*ones(n1,n2,n3)
	C  =∞*ones(n1,n2,n3)
	D  =∞*ones(n1,n2,n3)
	E  =∞*ones(n1,n2,n3)
	F  =∞*ones(n1,n2,n3)
	for i in 1:1:n1
        for j in 1:1:n2
            for k in 1:1:n3
				r=sqrt(((i-1)*dx-.5)^2+((j-1)*dy-.5)^2+((k-1)*dz-.5)^2)
                if (r-ϵ)< radius+1e-6 && (r+ϵ)> radius-1e-6
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

    for i in 1:1:n1
        for j in 1:1:n2
            for k in 1:1:n3
				r=sqrt(((i-1)*dx-.5)^2+((j-1)*dy-.5)^2+((k-1)*dz-.5)^2)
                if (r-ϵ)< radius+1e-6 && (r+ϵ)> radius-1e-6

                else
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
            if X[i,j,k]<1000
                if (X[i+1,j,k]==1000 || X[i-1,j,k]==1000 || X[i,j+1,k]==1000 || X[i,j-1,k]==1000||
                    X[i,j,k+1]==1000 || X[i,j,k-1]==1000)
                    edge[i,j,k]=true;
                end
            end
            end
        end
    end


	#set intial point
	fxy=sqrt(.4^2-(.3-.5)^2-(.3-.5)^2)+.5
	T=[.3,.3,fxy]

	nT=(T-[.5,.5,.5])/norm(T-[.5,.5,.5])

    epsi=.05
    init_point_target!(Uvals, initial,X, dx,dy,dz, nT, T,epsi,U,V,W)


    σx=0;
    σy=0;
    σz=0;
    H=(i,j,k,p,q,r)->sqrt(A[i,j,k]*p^2+B[i,j,k]*q^2+C[i,j,k]*r^2+D[i,j,k]*p*q+E[i,j,k]*q*r+F[i,j,k]*p*r)


    for i=1:nitrs
        LxFSweep.laxsweep(Uvals, status, edge, initial, X,  dx,dy,dz,σx,σy,σz, H, S1,S2, U,V,W,ϵ)

    end

	Linf=0
    Linf=ErrorComp.maxerror(N,Uvals,U, V, W,nT)
	L1=ErrorComp.L1error(N, Uvals, U, V, W, ϵ,radius,nT)

    return Linf,L1,Uvals
end

function mainDPGexact(N,nitrs) #solution when we use exact closest point mapping and exact singular values

    dx=1/(N-1)
	radius=.4
    U,V,W=CPmap.compUVWexactSphere(N,N,N,radius)
    ϵ=4*dx


    dx=dy=dz=1/(N-1)

    n1=n2=n3=N

    ∞ = 1000.0


    Uvals  =fill(∞, (n1, n2, n3))




    status=fill(true, (n1, n2,n3))
    edge=fill(false, (n1, n2,n3))
    initial=fill(false, (n1, n2,n3))
    X = ones(n1,n2,n3)

    for i in 1:1:n1
        for j in 1:1:n2
            for k in 1:1:n3
                r=sqrt(((i-1)*dx-.5)^2+((j-1)*dy-.5)^2+((k-1)*dz-.5)^2)
                if (r-ϵ)< radius+1e-6 && (r+ϵ)> radius-1e-6
                    X[i,j,k]=1-(r-radius)/r

                else
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
	            if X[i,j,k]<1000
	                if (X[i+1,j,k]==1000 || X[i-1,j,k]==1000 || X[i,j+1,k]==1000 || X[i,j-1,k]==1000||
	                    X[i,j,k+1]==1000 || X[i,j,k-1]==1000)
	                    edge[i,j,k]=true;
	                end
	            end
            end
        end
    end
	#target point
	fxy=sqrt(.4^2-(.1-.5)^2-(.5-.5)^2)+.5
	T=[.1,.5,fxy]
	r=norm(T-[.5,.5,.5])



	nT=(T-[.5,.5,.5])/norm(T-[.5,.5,.5])

    epsi=.05

    init_point_target!(Uvals, initial,X, dx,dy,dz, nT, T,epsi,U,V,W)

    testsig=.9
	println("sigma= ", testsig)
    σx=testsig
    σy=testsig
    σz=testsig

	H=(i,j,k,p,q,r)->sqrt(p^2+q^2+r^2)


    for i=1:nitrs
        #println("iteration=", i, " ")
        LxFSexactDPG.laxsweep(Uvals, status, edge, initial, X,  dx,dy,dz,σx,σy,σz, H, U,V,W,ϵ)

    end
    Linf=0
    Linf=ErrorComp.maxerror(N,Uvals,U, V, W,nT)
	L1=ErrorComp.L1error(N, Uvals, U, V, W, ϵ,radius,nT)


    return Linf,L1,Uvals
end



function mainPGMemSap(N,nitrs,Q)

    dx=1/(N-1)
	ϵ=2*(dx)^(.7)
	radius=.4
    U,V,W=CPmap.compUVWexactSphere(N,N,N,radius)

    #S1, S2 ,T11, T12, T13, T21, T22, T23,N1, N2, N3=DPG.CPM_info(Q, ϵ, dx, U, V, W)

    dx=dy=dz=1/(N-1)

    n1=n2=n3=N

    ∞ = 1000.0


    Uvals  =fill(∞, (n1, n2, n3))




    status=fill(true, (n1, n2,n3))
    edge=fill(false, (n1, n2,n3))
    initial=fill(false, (n1, n2,n3))
    X = ones(n1,n2,n3)
    #u=.4;
    #l=.2;
    for i in 1:1:n1
        for j in 1:1:n2
            for k in 1:1:n3
				r=sqrt(((i-1)*dx-.5)^2+((j-1)*dy-.5)^2+((k-1)*dz-.5)^2)
                if (r-ϵ)< radius+1e-6 && (r+ϵ)> radius-1e-6
                    #X[i,j,k]=1-(r-.1)/r

                else
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
				if X[i,j,k]<1000
	                if (X[i+1,j,k]==1000 || X[i-1,j,k]==1000 || X[i,j+1,k]==1000 || X[i,j-1,k]==1000||
	                    X[i,j,k+1]==1000 || X[i,j,k-1]==1000)
	                    edge[i,j,k]=true;
	                end
	            end
            end
        end
    end


	fxy=sqrt(.4^2-(.1-.5)^2-(.5-.5)^2)+.5
	T=[.1,.5,fxy]
	r=norm(T-[.5,.5,.5])



	nT=(T-[.5,.5,.5])/norm(T-[.5,.5,.5])

	epsi=.05
    LxFSweepMemSap.init_point_target!(Uvals, initial,X, dx,dy,dz, nT, T,epsi)


    testsig=.9
    σx=testsig
    σy=testsig
    σz=testsig
    H=(i,j,k,p,q,r)->sqrt(p^2+q^2+r^2)


    for i=1:nitrs
        LxFSweepMemSap.laxsweep(Uvals, status, edge, initial, X,  dx,dy,dz,σx,σy,σz, H)

    end
    Linf=0
    Linf=ErrorComp.maxerrorInterp(N,Uvals,U, V, W,nT)

    return Linf,Uvals
end









#point cloud for sphere
mf= MatFile("Qp.mat");
Q=get_variable(mf, "Q");
close(mf)

gridsize=[101,201,301,401,501,601,701,801]


  N=1

#@time Linf,L1,Uvals=mainDPGexact(gridsize[N],1)

#@time Linf,L1,Uvals=mainPGest(gridsize[N],20,Q)#,U,V,W,edge,status
@time Linf,Uvals=mainPGMemSap(gridsize[N],20,Q)
#println("For N= ", gridsize[N], " Linf= ", Linf, " ")
#println("For N= ", gridsize[N], " L1= ", L1, " ")
