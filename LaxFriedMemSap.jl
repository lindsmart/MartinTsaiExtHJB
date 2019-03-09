module LxFSweepMemSap

using Interpolations
using NearestNeighbors
using LinearAlgebra



function init_point_target!(u::Array{Float64}, s::Array{Bool},X::Array{Float64}, dx,dy,dz, nT, T,epsi)

	i=(floor(Int,(T[1])/dx))+1;
	j=(floor(Int,(T[2])/dy))+1;
	k=(floor(Int,(T[3])/dz))+1;


	iminus=(floor(Int,(T[1])/dx))+1;
	iplus =iminus+1;

	jminus=(floor(Int,(T[2])/dy))+1;
	jplus=jminus+1;

	kminus=(floor(Int,(T[3])/dz))+1;
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
			#ind,d=LxFSweep.kClosestPointsandDists(Ttree,xyz,Ttilde,1)

			u[i,j,k]=norm(T-xyz)#/X[i,j,k]#ErrorComp.exactdist(U[i,j,k],V[i,j,k],W[i,j,k],nT)#
			s[i,j,k]=true
		end
	end


		#u[i,j,k]=0.0
		#s[i,j,k]=true

end




function laxsweep( u::Array{Float64}, status::Array{Bool}, edge::Array{Bool},  initial::Array{Bool}, Rhs::Array{Float64},
	            dx,dy,dz,σx,σy,σz, H::Function)

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

			if edge[i,j,k]
				if status[i-1,j,k]==false
					u0=max(2*u[i+1,j,k]-u[i+2,j,k],u[i+2,j,k])
					u[i,j,k]=min(u0,u[i,j,k])
				end
				if status[i+1,j,k]==false
					u0=max(2*u[i-1,j,k]-u[i-2,j,k],u[i-2,j,k])
					u[i,j,k]=min(u0,u[i,j,k])
				end
				if status[i,j-1,k]==false
					u0=max(2*u[i,j+1,k]-u[i,j+2,k],u[i,j+2,k])
					u[i,j,k]=min(u0,u[i,j,k])
				end
				if status[i,j+1,k]==false
					u0=max(2*u[i,j-1,k]-u[i,j-2,k],u[i,j-2,k])
					u[i,j,k]=min(u0,u[i,j,k])
				end
				if status[i,j,k-1]==false
					u0=max(2*u[i,j,k+1]-u[i,j,k+2],u[i,j,k+2])
					u[i,j,k]=min(u0,u[i,j,k])
				end
				if status[i,j,k+1]==false
					u0=max(2*u[i,j,k-1]-u[i,j,k-2],u[i,j,k-2])
					u[i,j,k]=min(u0,u[i,j,k])
				end



			elseif status[i,j,k]==true && initial[i,j,k]==false  && edge[i,j,k]==false

				c=1/(σx/dx+σy/dy+σz/dz)
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
				uo=c*(rhs-H(i,j,k,pm,qm,r_m)+σx*pp+σy*qp+σz*rp)
				if uo<u[i,j,k]
					u[i,j,k]=uo
				end
			end
		end
	end


end

end
