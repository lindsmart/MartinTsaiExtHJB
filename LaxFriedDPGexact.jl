module LxFSexactDPG

using Interpolations
using NearestNeighbors
using LinearAlgebra

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

function exactdist(x,y,z)

  r=sqrt((x-.5).^2+(y-.5).^2+(z-.5).^2);
  Nxyz1=(x-.5)./r;
  Nxyz2=(y-.5)./r;
  Nxyz3=(z-.5)./r;
  Nxyz=[Nxyz1,Nxyz2,Nxyz3]
  N=[1,0,0]
  aa = cross(Nxyz, N);
  bb = dot(Nxyz',N');
  aa=norm(aa)
  d1 = r*atan(aa, bb)


end

function ghostnodeval(i,j,k, u::Array{Float64}, dx::Float64, dy::Float64,dz::Float64,
						U::Array{Float64},V::Array{Float64},W::Array{Float64},ϵ)
	gridpoint=[(i-1)*dx,(j-1)*dy,(k-1)*dz]

	Px=U[i,j,k]
	Py=V[i,j,k]
	Pz=W[i,j,k]


	normal=(gridpoint-[Px,Py,Pz])/norm(gridpoint-[Px,Py,Pz])

	alpha=(ϵ-2*sqrt(3)*dx);

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



	g=(i,j,k)->u[i,j,k]
	Ainterp = [g(i,j,k) for i in A_i, j in A_j, k in A_k]
	#if maximum(Ainterp)==1000
	#	println(i," ", j, " ", k, " ")
	#end
	itp = interpolate(Ainterp, BSpline(Cubic(InPlace(OnGrid()))))
	sitp = scale(itp, A_x1, A_x2, A_x3)

	uval=sitp(x,y,z)




	return uval

end



function laxsweep( u::Array{Float64}, status::Array{Bool}, edge::Array{Bool},  initial::Array{Bool}, Rhs::Array{Float64},
	            dx,dy,dz,σx,σy,σz, H::Function, U::Array{Float64},
				V::Array{Float64}, W::Array{Float64}, ϵ )

	pp,qp,rp=0.0, 0.0, 0.0
	pm,qm,r_m=0.0, 0.0, 0.0
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
