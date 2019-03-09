using Convex, SCS
using Optim
if true
    include("./CPmap.jl")
    include("./DPG.jl")
    include("./LaxFriedFS.jl")
    include("./LaxFriedDPGexact.jl")
    include("./LaxFriedMemSap.jl")
    include("./ErrorComp.jl")
end
function gss(H::Function,a,b,tol) #golden search section to perform the minimization in the Hamiltonian


    invphi = (sqrt(5) - 1)/2
    invphi2 = (3 - sqrt(5))/2

    a=min(a,b)
    b=max(a,b)
    h = b - a
    if h <= tol
         return a,b
     end

    # required steps to achieve tolerance
    n = convert(Int64,(ceil(log(tol/h)/log(invphi))))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = H(c)
    yd = H(d)
    xrange=range(1, stop=n-1, length=n-1)

    for k=1:27
        if yc < yd
            b = d
            d = c
            yd = yc
            h = invphi*h
            c = a + invphi2 * h
            yc = H(c)
        else
            a = c
            c = d
            yc = yd
            h = invphi*h
            d = a + invphi * h
            yd = H(d)
        end
    end

    if yc < yd
        return (a+d)/2
    else
        return (c+b)/2
    end
end




function laxsweepani( u::Array{Float64}, status::Array{Bool}, edge::Array{Bool},  initial::Array{Bool}, Rhs::Array{Float64},
	            dx,dy,dz, S1::Array{Float64}, S2::Array{Float64}, U::Array{Float64},
				V::Array{Float64}, W::Array{Float64}, ϵ ,b, T11::Array{Float64},T12::Array{Float64},
				T13::Array{Float64},T21::Array{Float64},T22::Array{Float64},T23::Array{Float64},
				control1::Array{Float64},control2::Array{Float64},control3::Array{Float64},fvals::Array{Float64})#,
				#A::Array{Float64},B::Array{Float64},C::Array{Float64},D::Array{Float64},E::Array{Float64},
				#F::Array{Float64})
	σx,σy,σz=0.0,0.0,0.0
	pp,qp,rp=0.0, 0.0, 0.0
	pm,qm,rm=0.0, 0.0, 0.0
	rhs = 0.0
	uw, ue, us, un, ut, ub, uo = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny, nz = map(dim->size(u, dim), (1,2,3))


   	for sgn1 in (-1,1), sgn2 in (-1,1), sgn3 in (-1,1)
		for k= (sgn3<0 ? nz-1 : 2): sgn3 : (sgn3<0 ? 2 : nz-1),
			j= (sgn2<0 ? ny-1 : 2): sgn2 : (sgn2<0 ? 2 : ny-1),
			i= (sgn1<0 ? nx-1 : 2): sgn1 : (sgn1<0 ? 2 : nx-1)
			if edge[i,j,k]==true && initial[i,j,k]==false
				uo=LxFSweep.ghostnodeval(i,j,k,u,dx,dy,dz,U,V,W,ϵ)
				if uo<u[i,j,k]
					u[i,j,k]=uo
				end

			elseif status[i,j,k]==true && initial[i,j,k]==false# && edge[i,j,k]==false
				gridpoint=[(i-1)*dx, (j-1)*dy,(k-1)*dz]
				η=norm(gridpoint-[U[i,j,k],V[i,j,k],W[i,j,k]])
				σ1=S1[i,j,k]
				σ2=S2[i,j,k]
				k1eta=(1-σ1)/η
				k1=k1eta/(1+η*k1eta)
				k2eta=(1-σ2)/η
				k2=k2eta/(1+η*k2eta)

				rhs=Rhs[i,j,k]

				uw=u[i-1,j,k];ue=u[i+1,j,k];
				us=u[i,j-1,k];un=u[i,j+1,k];
				ub=u[i,j,k-1];ut=u[i,j,k+1];

				pp=(ue+uw)/(2*dx)
				qp=(un+us)/(2*dy)
				rp=(ut+ub)/(2*dz)
				pm=(ue-uw)/(2*dx)
				qm=(un-us)/(2*dy)
				r_m=(ut-ub)/(2*dz)



				H=(t)->[pm,qm,r_m]'*((1/σ1)*cos(t)*[T11[i,j,k],T12[i,j,k],T13[i,j,k]]+
				(1/σ2)*sin(t)*[T21[i,j,k],T22[i,j,k],T23[i,j,k]])*exp(-b*abs(k1*(cos(t))^2+k2*(sin(t))^2))
				tmin=gss(H,0,2*pi,1e-5)

				Heval=H(tmin)
				#H=(i,j,k,p,q,r)->sqrt(A[i,j,k]*p^2+B[i,j,k]*q^2+C[i,j,k]*r^2+D[i,j,k]*p*q+E[i,j,k]*q*r+F[i,j,k]*p*r)
				#Heval=H(i,j,k,pm,qm,r_m)
				maxsing=min(S1[i,j,k],S2[i,j,k])
				maxsing=1/maxsing
				maxeig=max(maxsing,1) #or mu
				maxeig=maxeig

			#	σx=maxeig #sqrt(maxsig1sq+maxsig2sq+1)
			#	σy=maxeig #sqrt(maxsig1sq+maxsig2sq+1)
			#	σz=maxeig #sqrt(maxsig1sq+maxsig2sq+1)

				σx=(1/σ1+1/σ2)*.5#sqrt(maxsig1sq+maxsig2sq+1)
				σy=(1/σ1+1/σ2)*.5#sqrt(maxsig1sq+maxsig2sq+1)
				σz=(1/σ1+1/σ2)*.5#sqrt(maxsig1sq+maxsig2sq+1)
				c=1/(σx/dx+σy/dy+σz/dz)



				uo=c*(rhs+Heval+σx*pp+σy*qp+σz*rp)
				if uo<u[i,j,k]
					u[i,j,k]=uo
					control=((1/σ1)*cos(tmin)*[T11[i,j,k],T12[i,j,k],T13[i,j,k]]+
					(1/σ2)*sin(tmin)*[T21[i,j,k],T22[i,j,k],T23[i,j,k]])*exp(-b*abs(k1*(cos(tmin))^2+k2*(sin(tmin))^2))
					control1[i,j,k]=control[1]
					control2[i,j,k]=control[2]
					control3[i,j,k]=control[3]
					#fvals[i,j,k]=exp(-b*abs(k1*(cos(tmin))^2+k2*(sin(tmin))^2))
				end
			end
		end
	end


end

function mainAni(N,nitrs,Q::Array{Float64},nT1,T1,b);


	dx=1/(N-1)
    U,V,W=CPmap.compUVW(N,N,N,Q,7*dx)
    ϵ=4*dx
    S1, S2 ,T11, T12, T13, T21, T22, T23,N1, N2, N3=DPG.CPM_info(Q,ϵ, dx, U, V, W)
	fvals=zeros(N,N,N)

	dx=dy=dz=1/(N-1)

    n1=n2=n3=N

	control1=1000*ones(N,N,N)
	control2=1000*ones(N,N,N)
	control3=1000*ones(N,N,N)


	∞ = 1000.0



	Uvals  =fill(∞, (n1, n2, n3))

	A  =ones(n1,n2,n3)
	B  =ones(n1,n2,n3)
	C  =ones(n1,n2,n3)
	D  =ones(n1,n2,n3)
	E  =ones(n1,n2,n3)
	F  =ones(n1,n2,n3)
	#=for i in 1:1:n1
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
	end=#



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



	epsi=.04
	LxFSweep.init_point_target!(Uvals, initial,X, dx,dy,dz, nT, T,epsi)
	#LxFSweep.init_point_target2!(Uvals, initial,X, dx,dy,dz, nT1, T1,nT2,T2,epsi)

	#hamiltonian for eikonal case
	#=maxsig1sq=maximum(1 ./(S1))
	maxsig2sq=maximum(1 ./(S2))
	#maxeig=max(maxsig1sq,maxsig2sq)
	maxeig=max(maxsig1sq,1) #or mu
	σx=0;#maxeig #sqrt(maxsig1sq+maxsig2sq+1)
	σy=0;#maxeig #sqrt(maxsig1sq+maxsig2sq+1)
	σz=0;#maxeig #sqrt(maxsig1sq+maxsig2sq+1)
	H=(i,j,k,p,q,r)->sqrt(A[i,j,k]*p^2+B[i,j,k]*q^2+C[i,j,k]*r^2+D[i,j,k]*p*q+E[i,j,k]*q*r+F[i,j,k]*p*r)=#

	##hamiltonion for eikonal case euclidean space
	#σx=1
	#σy=1
	#σz=1
	#H=(i,j,k,p,q,r)->sqrt(p^2+q^2+r^2)


	for i=1:nitrs
		laxsweepani(Uvals, status, edge, initial, X,  dx,dy,dz, S1,S2, U,V,W,ϵ,b, T11,T12,T13,T21,
		T22,T23,control1,control2,control3,fvals)
	end



	return Uvals,control1, control2,control3#, T11,T12,T13,T21,T22,T23
end

mf= MatFile("QBunny3.mat");
Q=get_variable(mf, "QBunny3");

nT1=get_variable(mf, "nT2Bunny")
T1=get_variable(mf, "T2Bunny")

#close(mf)
@time Uvals,control1,control2,control3=mainAni(101,80,Q,nT1,T1,0.0);
