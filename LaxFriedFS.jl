module LxFSweep

using Interpolations
using NearestNeighbors
using LinearAlgebra
function kClosestPointsandDists(kdtree, gridpoint::Array{Float64,1}, surfdata::Array{Float64,2}, k)
    #kdtree = KDTree(surfdata)
    idxs, dists = knn(kdtree, gridpoint, 1, true);


    idxs, dists = knn(kdtree, surfdata[:,idxs[1]], k, true);
    #CPs=zeros(3,0)

    #for i in 1:size(idxs)[1]
    #     CPs=hcat(CPs,surfdata[:,idxs[i]])
    #end
return idxs,dists
end

#initilization for 2 point tests
function init_point_target2!(u::Array{Float64}, s::Array{Bool},X::Array{Float64}, dx,dy,dz, nT1, T1, nT2,T2,epsi)
	N=size(u,1)
	Ttilde=zeros(3,1001)
	dT=2*epsi/1000
	for i=0:1000
			alpha=i/1000
			Ttilde[:,i+1]=alpha*(T1+epsi*nT1)+(1-alpha)*(T1-epsi*nT1)
	end
	Ttree = KDTree(Ttilde)
	for a=0:1000

		x=Ttilde[1,a+1]
		y=Ttilde[2,a+1]
		z=Ttilde[3,a+1]
		iminus=(floor(Int,(x)/dx))+1;
		iplus =iminus+1;

		jminus=(floor(Int,(y)/dy))+1;
		jplus=jminus+1;

		kminus=(floor(Int,(z)/dz))+1;
		kplus=kminus+1;

		NBRS=[iminus jminus kminus;
		iminus jminus kplus;
		iminus jplus kminus;
		iminus jplus kplus;
		iplus jminus kminus;
		iplus jminus kplus;
		iplus jplus kminus;
		iplus jminus kplus;
		iplus jplus kplus]
		for n=1:8
			currPt=NBRS[n,:]
			i=currPt[1]
			j=currPt[2]
			k=currPt[3]
			if s[i,j,k]==false
				xyz=[(i-1)*dx ,(j-1)*dy ,(k-1)*dz]
				ind,d=kClosestPointsandDists(Ttree,xyz,Ttilde,1)
				u[i,j,k]=norm(Ttilde[:,ind[1]]-xyz)/X[i,j,k]
				s[i,j,k]=true
			end
		end
	end

	for i=0:1000
			alpha=i/1000
			Ttilde[:,i+1]=alpha*(T2+epsi*nT2)+(1-alpha)*(T2-epsi*nT2)
	end
	Ttree = KDTree(Ttilde)
	for a=0:1000

		x=Ttilde[1,a+1]
		y=Ttilde[2,a+1]
		z=Ttilde[3,a+1]
		iminus=(floor(Int,(x)/dx))+1;
		iplus =iminus+1;

		jminus=(floor(Int,(y)/dy))+1;
		jplus=jminus+1;

		kminus=(floor(Int,(z)/dz))+1;
		kplus=kminus+1;

		NBRS=[iminus jminus kminus;
		iminus jminus kplus;
		iminus jplus kminus;
		iminus jplus kplus;
		iplus jminus kminus;
		iplus jminus kplus;
		iplus jplus kminus;
		iplus jminus kplus;
		iplus jplus kplus]
		for n=1:8
			currPt=NBRS[n,:]
			i=currPt[1]
			j=currPt[2]
			k=currPt[3]
			if s[i,j,k]==false
				xyz=[(i-1)*dx ,(j-1)*dy ,(k-1)*dz]
				ind,d=kClosestPointsandDists(Ttree,xyz,Ttilde,1)
				u[i,j,k]=norm(Ttilde[:,ind[1]]-xyz)/X[i,j,k]
				s[i,j,k]=true
			end
		end
	end
end

function init_point_target!(u::Array{Float64}, s::Array{Bool},X::Array{Float64}, dx,dy,dz, nT, T,epsi)
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
		iplus =iminus+1;

		jminus=(floor(Int,(y)/dy))+1;
		jplus=jminus+1;

		kminus=(floor(Int,(z)/dz))+1;
		kplus=kminus+1;

		NBRS=[iminus jminus kminus;
		iminus jminus kplus;
		iminus jplus kminus;
		iminus jplus kplus;
		iplus jminus kminus;
		iplus jminus kplus;
		iplus jplus kminus;
		iplus jminus kplus;
		iplus jplus kplus]
		for n=1:8
			currPt=NBRS[n,:]
			i=currPt[1]
			j=currPt[2]
			k=currPt[3]
			if s[i,j,k]==false
				xyz=[(i-1)*dx ,(j-1)*dy ,(k-1)*dz]
				ind,d=kClosestPointsandDists(Ttree,xyz,Ttilde,1)
				u[i,j,k]=norm(Ttilde[:,ind[1]]-xyz)/X[i,j,k]
				s[i,j,k]=true
			end
		end
	end
end
function ghostnodeval(i,j,k, u::Array{Float64}, dx::Float64, dy::Float64,dz::Float64,
						U::Array{Float64},V::Array{Float64},W::Array{Float64},ϵ)
	gridpoint=[(i-1)*dx,(j-1)*dy,(k-1)*dz]

	Px=U[i,j,k]
	Py=V[i,j,k]
	Pz=W[i,j,k]

	normal=(gridpoint-[Px,Py,Pz])/norm(gridpoint-[Px,Py,Pz])
	x=Px+(ϵ-2*sqrt(3)*dx)*normal[1]
	y=Py+(ϵ-2*sqrt(3)*dy)*normal[2]
	z=Pz+(ϵ-2*sqrt(3)*dz)*normal[3]

	ibottom=(floor(Int,(x)/dx))+1-1
	jbottom=(floor(Int,(y)/dy))+1-1
	kbottom=(floor(Int,(z)/dz))+1-1
	xbottom=(ibottom-1)*dx
	ybottom=(jbottom-1)*dy
	zbottom=(kbottom-1)*dz
	if xbottom>500
		println(i," ", j," ", k, " ")
	end
	if ybottom>500
		println(i," ", j," ", k, " ")
	end
	if zbottom>500
		println(i," ", j," ", k, " ")
	end
	A_i = ibottom:(ibottom+3)
	A_j = jbottom:(jbottom+3)
	A_k = kbottom:(kbottom+3)
	A_x1 = xbottom:dx:(xbottom+3*dx)
	A_x2 = ybottom:dy:(ybottom+3*dy)
	A_x3 = zbottom:dz:(zbottom+3*dz)

	g=(i,j,k)->u[i,j,k]
	Ainterp = [g(i,j,k) for i in A_i, j in A_j, k in A_k]
	itp = interpolate(Ainterp, BSpline(Cubic(InPlace(OnGrid()))))
	sitp = scale(itp, A_x1, A_x2, A_x3)

	uval=sitp(x,y,z)
	if uval>=500
		uval=1000;
	end



	return uval

end



function laxsweep( u::Array{Float64}, status::Array{Bool}, edge::Array{Bool},  initial::Array{Bool}, Rhs::Array{Float64},
	            dx,dy,dz,σx,σy,σz, H::Function, S1::Array{Float64}, S2::Array{Float64}, U::Array{Float64},
				V::Array{Float64}, W::Array{Float64}, ϵ )

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
				uo=ghostnodeval(i,j,k,u,dx,dy,dz,U,V,W,ϵ)
				if uo<u[i,j,k]
					u[i,j,k]=uo
				end

			elseif status[i,j,k]==true && initial[i,j,k]==false# && edge[i,j,k]==false
				maxsing=min(S1[i,j,k],S2[i,j,k])
				maxsing=1/maxsing
				maxeig=max(maxsing,1) #or mu
				maxeig=1*maxeig

				σx=maxeig #sqrt(maxsig1sq+maxsig2sq+1)
				σy=maxeig#sqrt(maxsig1sq+maxsig2sq+1)
				σz=maxeig#sqrt(maxsig1sq+maxsig2sq+1)
				c=1/(σx/dx+σy/dy+σz/dz)
				rhs=Rhs[i,j,k]
				#=if edge[i,j,k]==true
					if status[i-1,j,k]==false #&& u[i-1,j,k]>500
						uw=ghostnodeval(i-1,j,k, u, dx, dy,dz, U, V, W,ϵ)
					else
						uw=u[i-1,j,k];
					end
					if status[i+1,j,k]==false# && u[i+1,j,k]>500
						ue=ghostnodeval(i+1,j,k, u, dx, dy,dz, U, V, W,ϵ)
					else
						ue=u[i+1,j,k];
					end
					if status[i,j-1,k]==false #&& u[i,j-1,k]>500
						us=ghostnodeval(i,j-1,k, u, dx, dy,dz, U, V, W,ϵ)
					else
						us=u[i,j-1,k];
					end
					if status[i,j+1,k]==false #&& u[i,j+1,k]>500
						un=ghostnodeval(i,j+1,k, u, dx, dy,dz, U, V, W,ϵ)
					else
						 un=u[i,j+1,k];
					end
					if status[i,j,k-1]==false #&& u[i,j,k-1]>500
						ub=ghostnodeval(i,j,k-1, u, dx, dy,dz, U, V, W,ϵ)
					else
						 ub=u[i,j,k-1];
					end
					if status[i,j,k+1]==false #&& u[i,j,k+1]>500
						ut=ghostnodeval(i,j,k+1, u, dx, dy,dz, U, V, W,ϵ)
					else
						 ut=u[i,j,k+1];
					end
				else=#
					uw=u[i-1,j,k];ue=u[i+1,j,k];
					us=u[i,j-1,k];un=u[i,j+1,k];
					ub=u[i,j,k-1];ut=u[i,j,k+1];
				#end
				pp=(ue+uw)/(2*dx)
				qp=(un+us)/(2*dy)
				rp=(ut+ub)/(2*dz)
				pm=(ue-uw)/(2*dx)
				qm=(un-us)/(2*dy)
				r_m=(ut-ub)/(2*dz)
				uo=c*(rhs-H(i,j,k,pm,qm,r_m)+σx*pp+σy*qp+σz*rp)
				if uo<u[i,j,k]
					u[i,j,k]=uo
				end
			end
		end
	end


end

end
