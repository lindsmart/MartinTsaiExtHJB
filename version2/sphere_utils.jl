module SphereUtils
using Interpolations
using LinearAlgebra

function exact_dist(x,y,z,target_normal)

  r=sqrt((x-.5).^2+(y-.5).^2+(z-.5).^2);
  Nxyz1=(x-.5)./r;
  Nxyz2=(y-.5)./r;
  Nxyz3=(z-.5)./r;
  Nxyz=[Nxyz1,Nxyz2,Nxyz3]
  aa = cross(Nxyz, target_normal);
  bb = dot(Nxyz',target_normal');
  aa=norm(aa)
  d1 = r*atan(aa, bb)


end

function init_point_target!(uvals::Array{Float64},
	 						status::Array{Bool},
							dx,
							dy,
							dz,
							target_point,
							target_normal,
							tube_eps,
							U::Array{Float64},
							V::Array{Float64},
							W::Array{Float64})

	xyz_extend=zeros(3,1001)

	for i=0:1000
			alpha=i/1000
			xyz_extend[:,i+1]=alpha*(target_point+tube_eps*target_normal)+(1-alpha)*(target_point-tube_eps*target_normal)
	end

	for a=0:1000
		x=xyz_extend[1,a+1]
		y=xyz_extend[2,a+1]
		z=xyz_extend[3,a+1]
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
			if status[i,j,k]
				uvals[i,j,k] = exact_dist(U[i,j,k],
										 V[i,j,k],
									     W[i,j,k],
									     target_normal)
			    status[i,j,k] = false
			end
		end
	end
end


function cp_map_on_sphere_exact(n1,n2,n3,radius)

    dx=1/(n1-1)
    dy=1/(n2-1)
    dz=1/(n3-1)
    U = zeros(n1,n2,n3)
    V = zeros(n1,n2,n3)
    W = zeros(n1,n2,n3)
    for I in CartesianIndices(U)
                gridpoint=[(I[1]-1)*dx,(I[2]-1)*dy,(I[3]-1)*dz]
                r=norm(gridpoint-[.5,.5,.5])
				t=radius/r
                CPx=t*(gridpoint[1]-.5)+.5;
				CPy=t*(gridpoint[2]-.5)+.5;
				CPz=t*(gridpoint[3]-.5)+.5;
                U[I[1],I[2],I[3]]=CPx
                V[I[1],I[2],I[3]]=CPy
                W[I[1],I[2],I[3]]=CPz
    end
return U,V,W

end

function Linterp(x::Float64, x1::Float64, x2::Float64, q00::Float64, q01::Float64)
	return ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
end


function triLinterp(x::Float64,  y::Float64,  z::Float64,  q000::Float64,  q001::Float64,
	  q010::Float64,  q011::Float64,  q100::Float64,  q101::Float64,  q110::Float64,
	   q111::Float64,  x1::Float64,  x2::Float64,  y1::Float64,  y2::Float64,  z1::Float64,  z2::Float64)
   x00 = Linterp(x, x1, x2, q000, q100);
   x10 = Linterp(x, x1, x2, q010, q110);
   x01 = Linterp(x, x1, x2, q001, q101);
   x11 = Linterp(x, x1, x2, q011, q111);
   r0 = Linterp(y, y1, y2, x00, x10);
   r1 = Linterp(y, y1, y2, x01, x11);

  return Linterp(z, z1, z2, r0, r1);
end


function ghost_node_eval(i,j,k, uvals::Array{Float64}, dx::Float64, dy::Float64,dz::Float64,
						U::Array{Float64},V::Array{Float64},W::Array{Float64},eps)
	gridpoint=[(i-1)*dx,(j-1)*dy,(k-1)*dz]

	Px=U[i,j,k]
	Py=V[i,j,k]
	Pz=W[i,j,k]

	normal=(gridpoint-[Px,Py,Pz])/norm(gridpoint-[Px,Py,Pz])
	alpha=(eps-2*sqrt(3)*dx);

	x=Px+alpha*normal[1]
	y=Py+alpha*normal[2]
	z=Pz+alpha*normal[3]
	#uncomment to use trilinear inteprolation instead of tricubic
	#=x=gridpoint[1]+(3*dx)*normal[1]
	y=gridpoint[2]+(3*dy)*normal[2]
	z=gridpoint[3]+(3*dz)*normal[3]=#

	#=iminus=(floor(Int,(x)/dx))+1
	jminus=(floor(Int,(y)/dy))+1
	kminus=(floor(Int,(z)/dz))+1
	iplus=iminus+1
	jplus=jminus+1
	kplus=kminus+1
	x1=(iminus-1)*dx
	y1=(jminus-1)*dy
	z1=(kminus-1)*dz
	x2=(iplus-1)*dx
	y2=(jplus-1)*dy
	z2=(kplus-1)*dz

	q000=u[iminus,jminus,kminus]
	q001=u[iminus,jminus,kplus]
	q010=u[iminus,jplus,kminus]
	q011=u[iminus,jplus,kplus]
	q100=u[iplus,jminus,kminus]
	q101=u[iplus,jminus,kplus]
	q110=u[iplus,jplus,kminus]
	q111=u[iplus,jplus,kplus]

	uval=triLinterp(x,  y,  z,  q000,  q001, q010,  q011,  q100,  q101,  q110,
		   q111,  x1,  x2,  y1,  y2,  z1,  z2)=#

	ibottom=(floor(Int,(x)/dx))+1-1
	jbottom=(floor(Int,(y)/dy))+1-1
	kbottom=(floor(Int,(z)/dz))+1-1
	xbottom=(ibottom-1)*dx
	ybottom=(jbottom-1)*dy
	zbottom=(kbottom-1)*dz

	A_i = ibottom:(ibottom+3)
	A_j = jbottom:(jbottom+3)
	A_k = kbottom:(kbottom+3)
	A_x1 = xbottom:dx:(xbottom+3*dx)
	A_x2 = ybottom:dy:(ybottom+3*dy)
	A_x3 = zbottom:dz:(zbottom+3*dz)

	g=(i,j,k)->uvals[i,j,k]
	Ainterp = [g(i,j,k) for i in A_i, j in A_j, k in A_k]

	itp = interpolate(Ainterp, BSpline(Cubic(InPlace(OnGrid()))))
	sitp = scale(itp, A_x1, A_x2, A_x3)

	uval=sitp(x,y,z)

	return uval
end

function lax_sweep(uvals::Array{Float64},
				  status::Array{Bool},
				  edge::Array{Bool},
				  rhs::Array{Float64},
	              dx,
				  dy,
				  dz,
				  σx,
				  σy,
				  σz,
				  H::Function,
				  U::Array{Float64},
				  V::Array{Float64},
				  W::Array{Float64},
				  eps)

	pp,qp,rp=0.0, 0.0, 0.0
	pm,qm,r_m=0.0, 0.0, 0.0
	rhs_val = 0.0
	uw, ue, us, un, ut, ub, uo = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny, nz = map(dim->size(uvals, dim), (1,2,3))
   	for sgn1 in (-1,1), sgn2 in (-1,1), sgn3 in (-1,1)
		for k= (sgn3<0 ? nz-1 : 2): sgn3 : (sgn3<0 ? 2 : nz-1),
			j= (sgn2<0 ? ny-1 : 2): sgn2 : (sgn2<0 ? 2 : ny-1),
			i= (sgn1<0 ? nx-1 : 2): sgn1 : (sgn1<0 ? 2 : nx-1)

			if edge[i,j,k]
				uo=ghost_node_eval(i,j,k,uvals,dx,dy,dz,U,V,W,eps)
				if uo<uvals[i,j,k]
					uvals[i,j,k]=uo
				end

			elseif status[i,j,k]
				c=1/(σx/dx+σy/dy+σz/dz)
				rhs_val=rhs[i,j,k]

				uw=uvals[i-1,j,k];ue=uvals[i+1,j,k];
				us=uvals[i,j-1,k];un=uvals[i,j+1,k];
				ub=uvals[i,j,k-1];ut=uvals[i,j,k+1];

				pp=(ue+uw)/(2*dx)
				qp=(un+us)/(2*dy)
				rp=(ut+ub)/(2*dz)
				pm=(ue-uw)/(2*dx)
				qm=(un-us)/(2*dy)
				r_m=(ut-ub)/(2*dz)
				uo=c*(rhs_val-H(pm,qm,r_m)+σx*pp+σy*qp+σz*rp)
				if uo<uvals[i,j,k]
					uvals[i,j,k]=uo
				end
			end
		end
	end


end

function l1_error(uvals::Array{Float64},
				  U::Array{Float64},
				  V::Array{Float64},
				  W::Array{Float64},
				  dx,
				  dy,
				  dz,
				  eps,
				  radius,
				  target_normal)

	 L1err=0
	 for I in CartesianIndices(uvals)
		i = I[1]
		j = I[2]
		k = I[3]
	    if 0<uvals[i,j,k]<500 #&& initial[i,j,k]==false
	      CPx=U[i,j,k]
	      CPy=V[i,j,k]
	      CPz=W[i,j,k]
				gridpoint=[(i-1)*dx,(j-1)*dy,(k-1)*dz]
				r=norm(gridpoint-[.5,.5,.5])
				η=-1*(r-radius)
				σ1=1+η/r
				σ2=1+η/r
				Jac=σ1*σ2
				kern=(1+cos(pi*η/eps))/(2*eps)

				uexact=exact_dist(CPx,CPy,CPz,target_normal)
				uo=uvals[i,j,k]
				diff=abs(uo-uexact);
				L1err+=diff*dx*dy*dz*kern*Jac
			end
  	end
	return  L1err
end

function linf_error(uvals::Array{Float64},
					U::Array{Float64},
					V::Array{Float64},
					W::Array{Float64},
					target_normal)
	 maxdiff =0
	 ijk=[0,0,0]
	 for I in CartesianIndices(uvals)
		i = I[1]
		j = I[2]
		k = I[3]
	    if 0<uvals[i,j,k]<1000
			CPx=U[i,j,k]
			CPy=V[i,j,k]
			CPz=W[i,j,k]
			uexact=exact_dist(CPx,CPy,CPz,target_normal)
			uo=uvals[i,j,k]
			diff=abs(uo-uexact);
			if diff>maxdiff
				maxdiff=diff
				ijk=[i,j,k]
			end
		end
  	end
	return maxdiff
end

end
