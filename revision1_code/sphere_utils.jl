module SphereUtils
using Interpolations
using LinearAlgebra
using Printf

function exact_dist(x,y,z,target_normal)

  r=sqrt((x).^2+(y).^2+(z).^2);
  Nxyz=[x/r,y/r,z/r]
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

	xyz_extend=zeros(3,501)
	N = 2*Int(1/dx)+1
	for i=0:500
			alpha=i/500
			xyz_extend[:,i+1]=alpha*(target_point+tube_eps*target_normal)+(1-alpha)*(target_point-tube_eps*target_normal)
	end
	for a=0:500
		x=xyz_extend[1,a+1]
		y=xyz_extend[2,a+1]
		z=xyz_extend[3,a+1]
		iminus=(floor(Int,(x+1)/dx))+1;
		jminus=(floor(Int,(y+1)/dy))+1;
		kminus=(floor(Int,(z+1)/dz))+1;
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

function init_point_target_high_order!(uvals::Array{Float64},
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
	N = 2*Int(1/dx)+1
	for i=0:1000
			alpha=i/1000
			xyz_extend[:,i+1]=alpha*(target_point+tube_eps*target_normal)+(1-alpha)*(target_point-tube_eps*target_normal)
	end

	for a=0:1000
		x=xyz_extend[1,a+1]
		y=xyz_extend[2,a+1]
		z=xyz_extend[3,a+1]
		iminus=(floor(Int,(x+1)/dx))+1;
		jminus=(floor(Int,(y+1)/dy))+1;
		kminus=(floor(Int,(z+1)/dz))+1;
		A_i = (iminus-2):(iminus+3)
		A_j = (jminus-2):(jminus+3)
		A_k = (kminus-2):(kminus+3)

		NBRS = [[i,j,k] for i in A_i, j in A_j, k in A_k]

		for n=1:216
			currPt=NBRS[n]
			i=currPt[1]
			j=currPt[2]
			k=currPt[3]
			if max(i,j,k)<=N && min(i,j,k)>0
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
end

function init_point_target_high_order_v2!(uvals::Array{Float64},
 									  status::Array{Bool},
									  dx,
									  dy,
									  dz,
									  target_point,
									  target_normal,
									  U::Array{Float64},
									  V::Array{Float64},
									  W::Array{Float64})

	for I in CartesianIndices(uvals)
	   i = I[1]
	   j = I[2]
	   k = I[3]
	   CPx=U[i,j,k]
	   CPy=V[i,j,k]
	   CPz=W[i,j,k]
	   uexact=exact_dist(CPx,CPy,CPz,target_normal)
	   if uexact<.2
			if status[i,j,k]
				uvals[i,j,k] = uexact
			    status[i,j,k] = false
			end
		end
	end
end

function cp_map_on_sphere_exact(n1,n2,n3,radius)

    dx=2/(n1-1)
    dy=2/(n2-1)
    dz=2/(n3-1)
    U = zeros(n1,n2,n3)
    V = zeros(n1,n2,n3)
    W = zeros(n1,n2,n3)
	t=0.0
    for I in CartesianIndices(U)
                gridpoint=[(I[1]-1)*dx-1,(I[2]-1)*dy-1,(I[3]-1)*dz-1]
                r=norm(gridpoint)
				if r > 1e-12
					t=radius/r
				else
					t=0.0
				end

                CPx=t*(gridpoint[1]);
				CPy=t*(gridpoint[2]);
				CPz=t*(gridpoint[3]);
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
						U::Array{Float64},V::Array{Float64},W::Array{Float64},ϵ)
	gridpoint=[(i-1)*dx-1,(j-1)*dy-1,(k-1)*dz-1]

	Px=U[i,j,k]
	Py=V[i,j,k]
	Pz=W[i,j,k]

	normal=(gridpoint-[Px,Py,Pz])/norm(gridpoint-[Px,Py,Pz])
	alpha=(ϵ-2*sqrt(3)*dx);#0.0#

	x=Px+alpha*normal[1]
	y=Py+alpha*normal[2]
	z=Pz+alpha*normal[3]


	#=uncomment to use trilinear inteprolation instead of tricubic

	iminus=(floor(Int,(x)/dx))+1
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

	q000=uvals[iminus,jminus,kminus]
	q001=uvals[iminus,jminus,kplus]
	q010=uvals[iminus,jplus,kminus]
	q011=uvals[iminus,jplus,kplus]
	q100=uvals[iplus,jminus,kminus]
	q101=uvals[iplus,jminus,kplus]
	q110=uvals[iplus,jplus,kminus]
	q111=uvals[iplus,jplus,kplus]

	uval=triLinterp(x,  y,  z,  q000,  q001, q010,  q011,  q100,  q101,  q110,
		   q111,  x1,  x2,  y1,  y2,  z1,  z2)=#

	ibottom=floor(Int,(x+1)/dx)+1-1
	jbottom=floor(Int,(y+1)/dy)+1-1
	kbottom=floor(Int,(z+1)/dz)+1-1
	xbottom=(ibottom-1)*dx-1
	ybottom=(jbottom-1)*dy-1
	zbottom=(kbottom-1)*dz-1


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
				  ϵ)
	max_error = 0.0
	pp,qp,rp = 0.0, 0.0, 0.0
	pm,qm,r_m = 0.0, 0.0, 0.0
	rhs_val = 0.0
	uw, ue, us, un, ut, ub, uo = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny, nz = map(dim->size(uvals, dim), (1,2,3))

   	for sgn1 in (-1,1), sgn2 in (-1,1), sgn3 in (-1,1)
        for k= (sgn3<0 ? nz-1 : 2): sgn3 : (sgn3<0 ? 2 : nz-1),
			j= (sgn2<0 ? ny-1 : 2): sgn2 : (sgn2<0 ? 2 : ny-1),
			i= (sgn1<0 ? nx-1 : 2): sgn1 : (sgn1<0 ? 2 : nx-1)

			if edge[i,j,k]
				uo=ghost_node_eval(i,j,k,uvals,dx,dy,dz,U,V,W,ϵ)
				if uo<uvals[i,j,k]
					uvals[i,j,k]=uo
				end
				# if status[i-1,j,k]==false
				# 	u0=max(2*uvals[i+1,j,k]-uvals[i+2,j,k],uvals[i+2,j,k])
				# 	uvals[i,j,k]=min(u0,uvals[i,j,k])
				# end
				# if status[i+1,j,k]==false
				# 	u0=max(2*uvals[i-1,j,k]-uvals[i-2,j,k],uvals[i-2,j,k])
				# 	uvals[i,j,k]=min(u0,uvals[i,j,k])
				# end
				# if status[i,j-1,k]==false
				# 	u0=max(2*uvals[i,j+1,k]-uvals[i,j+2,k],uvals[i,j+2,k])
				# 	uvals[i,j,k]=min(u0,uvals[i,j,k])
				# end
				# if status[i,j+1,k]==false
				# 	u0=max(2*uvals[i,j-1,k]-uvals[i,j-2,k],uvals[i,j-2,k])
				# 	uvals[i,j,k]=min(u0,uvals[i,j,k])
				# end
				# if status[i,j,k-1]==false
				# 	u0=max(2*uvals[i,j,k+1]-uvals[i,j,k+2],uvals[i,j,k+2])
				# 	uvals[i,j,k]=min(u0,uvals[i,j,k])
				# end
				# if status[i,j,k+1]==false
				# 	u0=max(2*uvals[i,j,k-1]-uvals[i,j,k-2],uvals[i,j,k-2])
				# 	uvals[i,j,k]=min(u0,uvals[i,j,k])
				# end

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
				uo=c*(rhs_val-H(i,j,k,pm,qm,r_m)+σx*pp+σy*qp+σz*rp)
				if uo<uvals[i,j,k]
					curr_error = abs(uo - uvals[i,j,k])
					uvals[i,j,k]=uo
					max_error = max(curr_error, max_error)
				end
			end
		end
	end
	return max_error

end


function lax_sweep_point_cloud(uvals::Array{Float64},
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
							  S1::Array{Float64},
							  S2::Array{Float64},
							  ϵ)
	max_error = 0.0
	pp,qp,rp = 0.0, 0.0, 0.0
	pm,qm,r_m = 0.0, 0.0, 0.0
	rhs_val = 0.0
	uw, ue, us, un, ut, ub, uo = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny, nz = map(dim->size(uvals, dim), (1,2,3))

   	for sgn1 in (-1,1), sgn2 in (-1,1), sgn3 in (-1,1)
        for k= (sgn3<0 ? nz-1 : 2): sgn3 : (sgn3<0 ? 2 : nz-1),
			j= (sgn2<0 ? ny-1 : 2): sgn2 : (sgn2<0 ? 2 : ny-1),
			i= (sgn1<0 ? nx-1 : 2): sgn1 : (sgn1<0 ? 2 : nx-1)

			if edge[i,j,k]
				uo=ghost_node_eval(i,j,k,uvals,dx,dy,dz,U,V,W,ϵ)
				if uo<uvals[i,j,k]
					uvals[i,j,k]=uo
				end

			elseif status[i,j,k]
				maxsing=min(S1[i,j,k],S2[i,j,k])
				maxsing=1/maxsing
				maxeig=max(maxsing,1) #or mu
				# maxeig=1.16^2#*maxeig
				#
				σx=maxeig #sqrt(maxsig1sq+maxsig2sq+1)
				σy=maxeig#sqrt(maxsig1sq+maxsig2sq+1)
				σz=maxeig#sqrt(maxsig1sq+maxsig2sq+1)


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
				uo=c*(rhs_val-H(i,j,k,pm,qm,r_m)+σx*pp+σy*qp+σz*rp)
				if uo<uvals[i,j,k]
					curr_error = abs(uo - uvals[i,j,k])
					uvals[i,j,k]=uo
					max_error = max(curr_error, max_error)
				end
			end
		end
	end
	return max_error

end


function lax_sweep_point_cloud_high_order(uvals::Array{Float64},
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
										  S1::Array{Float64},
										  S2::Array{Float64},
										  ϵ)
	max_error = 0.0
	pp,qp,rp = 0.0, 0.0, 0.0
	pm,qm,r_m = 0.0, 0.0, 0.0
	rhs_val = 0.0
	pwm, qwm, rwm = 0.0, 0.0, 0.0
	pwp, qwp, rwp = 0.0, 0.0, 0.0
	prm, qrm, rrm = 0.0, 0.0, 0.0
	prp, qrp, rrp = 0.0, 0.0, 0.0

	p_p, p_m, q_p, q_m, r_p, r_m1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	uw1, ue1, us1, un1, ut1, ub1, uo = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	uw2, ue2, us2, un2, ut2, ub2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny, nz = map(dim->size(uvals, dim), (1,2,3))
   	for sgn1 in (-1,1), sgn2 in (-1,1), sgn3 in (-1,1)
		for k= (sgn3<0 ? nz-1 : 2): sgn3 : (sgn3<0 ? 2 : nz-1),
			j= (sgn2<0 ? ny-1 : 2): sgn2 : (sgn2<0 ? 2 : ny-1),
			i= (sgn1<0 ? nx-1 : 2): sgn1 : (sgn1<0 ? 2 : nx-1)

			if edge[i,j,k]
				uo=ghost_node_eval(i,j,k,uvals,dx,dy,dz,U,V,W,ϵ)
				# if uo<uvals[i,j,k]
				uvals[i,j,k]=uo
				# end
				# end

			elseif status[i,j,k]
				# gridpoint=[(i-1)*dx-1,(j-1)*dy-1,(k-1)*dz-1]
				# maxsing=min(S1[i,j,k],S2[i,j,k])
				# maxsing=1/maxsing
				# maxeig=max(maxsing,1) #or mu
				# maxeig=1#*maxeig
				#
				# σx=maxeig #sqrt(maxsig1sq+maxsig2sq+1)
				# σy=maxeig#sqrt(maxsig1sq+maxsig2sq+1)
				# σz=maxeig#sqrt(maxsig1sq+maxsig2sq+1)

				c=1/(σx/dx+σy/dy+σz/dz)
				rhs_val=rhs[i,j,k]

				uw1=uvals[i-1,j,k];ue1=uvals[i+1,j,k];
				us1=uvals[i,j-1,k];un1=uvals[i,j+1,k];
				ub1=uvals[i,j,k-1];ut1=uvals[i,j,k+1];

				uw2=uvals[i-2,j,k];ue2=uvals[i+2,j,k];
				us2=uvals[i,j-2,k];un2=uvals[i,j+2,k];
				ub2=uvals[i,j,k-2];ut2=uvals[i,j,k+2];

				weno_eps = 1.e-10

				prm = (weno_eps+(uvals[i,j,k]-2*uw1+uw2)^2)/(weno_eps+(ue1-2*uvals[i,j,k]+uw1)^2)
				pwm = 1.0/(1.0+2.0*prm^2)
				prp = (weno_eps+(ue2-2*ue1+uvals[i,j,k])^2)/(weno_eps+(ue1-2*uvals[i,j,k]+uw1)^2)
				pwp = 1.0/(1.0 + 2.0*prp^2)

				qrm = (weno_eps+(uvals[i,j,k]-2*us1+us2)^2)/(weno_eps+(un1-2*uvals[i,j,k]+us1)^2)
				qwm = 1.0/(1.0+2.0*qrm^2)
				qrp = (weno_eps+(un2-2*un1+uvals[i,j,k])^2)/(weno_eps+(un1-2*uvals[i,j,k]+us1)^2)
				qwp = 1.0/(1.0 + 2.0*qrp^2)

				rrm = (weno_eps+(uvals[i,j,k]-2*ub1+ub2)^2)/(weno_eps+(ut1-2*uvals[i,j,k]+ub1)^2)
				rwm = 1.0/(1.0+2.0*rrm^2)
				rrp = (weno_eps+(ut2-2*ut1+uvals[i,j,k])^2)/(weno_eps+(ut1-2*uvals[i,j,k]+ub1)^2)
				rwp = 1.0/(1.0 + 2.0*rrp^2)

				p_m = (1-pwm)*(ue1-uw1)/(2*dx) + pwm*((3*uvals[i,j,k]-4*uw1+uw2)/(2*dx))
				p_p = (1-pwp)*(ue1-uw1)/(2*dx) + pwp*((-1*ue2+4*ue1-3*uvals[i,j,k])/(2*dx))

				q_m = (1-qwm)*(un1-us1)/(2*dy) + qwm*((3*uvals[i,j,k]-4*us1+us2)/(2*dy))
				q_p = (1-qwp)*(un1-us1)/(2*dy) + qwp*((-1*un2+4*un1-3*uvals[i,j,k])/(2*dy))

				r_m1 = (1-rwm)*(ut1-ub1)/(2*dz) + rwm*((3*uvals[i,j,k]-4*ub1+ub2)/(2*dz))
				r_p = (1-rwp)*(ut1-ub1)/(2*dz) + rwp*((-1*ut2+4*ut1-3*uvals[i,j,k])/(2*dz))

				pp = (p_m+p_p)/2.0
				qp = (q_m+q_p)/2.0
				rp = (r_m1+r_p)/2.0

				pm = (p_p-p_m)/2.0
				qm = (q_p-q_m)/2.0
				r_m = (r_p-r_m1)/2.0


				uo=c*(rhs_val-H(i,j,k,pp,qp,rp)+σx*pm+σy*qm+σz*r_m)+uvals[i,j,k]

				curr_error = abs(uo - uvals[i,j,k])
				uvals[i,j,k]=uo
				max_error = max(curr_error, max_error)

			end
		end
	end
	return max_error

end


function lax_sweep_high_order(uvals::Array{Float64},
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
							  ϵ)
	max_error = 0.0
	pp,qp,rp = 0.0, 0.0, 0.0
	pm,qm,r_m = 0.0, 0.0, 0.0
	rhs_val = 0.0
	pwm, qwm, rwm = 0.0, 0.0, 0.0
	pwp, qwp, rwp = 0.0, 0.0, 0.0
	prm, qrm, rrm = 0.0, 0.0, 0.0
	prp, qrp, rrp = 0.0, 0.0, 0.0

	p_p, p_m, q_p, q_m, r_p, r_m1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	uw1, ue1, us1, un1, ut1, ub1, uo = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	uw2, ue2, us2, un2, ut2, ub2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny, nz = map(dim->size(uvals, dim), (1,2,3))
   	for sgn1 in (-1,1), sgn2 in (-1,1), sgn3 in (-1,1)
		for k= (sgn3<0 ? nz-1 : 2): sgn3 : (sgn3<0 ? 2 : nz-1),
			j= (sgn2<0 ? ny-1 : 2): sgn2 : (sgn2<0 ? 2 : ny-1),
			i= (sgn1<0 ? nx-1 : 2): sgn1 : (sgn1<0 ? 2 : nx-1)

			if edge[i,j,k]
				uo=ghost_node_eval(i,j,k,uvals,dx,dy,dz,U,V,W,ϵ)
				# if uo<uvals[i,j,k]
				uvals[i,j,k]=uo
				# end

			elseif status[i,j,k]


				c=1/(σx/dx+σy/dy+σz/dz)
				rhs_val=rhs[i,j,k]

				uw1=uvals[i-1,j,k];ue1=uvals[i+1,j,k];
				us1=uvals[i,j-1,k];un1=uvals[i,j+1,k];
				ub1=uvals[i,j,k-1];ut1=uvals[i,j,k+1];

				uw2=uvals[i-2,j,k];ue2=uvals[i+2,j,k];
				us2=uvals[i,j-2,k];un2=uvals[i,j+2,k];
				ub2=uvals[i,j,k-2];ut2=uvals[i,j,k+2];

				weno_eps = 1.e-10

				prm = (weno_eps+(uvals[i,j,k]-2*uw1+uw2)^2)/(weno_eps+(ue1-2*uvals[i,j,k]+uw1)^2)
				pwm = 1.0/(1.0+2.0*prm^2)
				prp = (weno_eps+(ue2-2*ue1+uvals[i,j,k])^2)/(weno_eps+(ue1-2*uvals[i,j,k]+uw1)^2)
				pwp = 1.0/(1.0 + 2.0*prp^2)

				qrm = (weno_eps+(uvals[i,j,k]-2*us1+us2)^2)/(weno_eps+(un1-2*uvals[i,j,k]+us1)^2)
				qwm = 1.0/(1.0+2.0*qrm^2)
				qrp = (weno_eps+(un2-2*un1+uvals[i,j,k])^2)/(weno_eps+(un1-2*uvals[i,j,k]+us1)^2)
				qwp = 1.0/(1.0 + 2.0*qrp^2)

				rrm = (weno_eps+(uvals[i,j,k]-2*ub1+ub2)^2)/(weno_eps+(ut1-2*uvals[i,j,k]+ub1)^2)
				rwm = 1.0/(1.0+2.0*rrm^2)
				rrp = (weno_eps+(ut2-2*ut1+uvals[i,j,k])^2)/(weno_eps+(ut1-2*uvals[i,j,k]+ub1)^2)
				rwp = 1.0/(1.0 + 2.0*rrp^2)

				p_m = (1-pwm)*(ue1-uw1)/(2*dx) + pwm*((3*uvals[i,j,k]-4*uw1+uw2)/(2*dx))
				p_p = (1-pwp)*(ue1-uw1)/(2*dx) + pwp*((-1*ue2+4*ue1-3*uvals[i,j,k])/(2*dx))

				q_m = (1-qwm)*(un1-us1)/(2*dy) + qwm*((3*uvals[i,j,k]-4*us1+us2)/(2*dy))
				q_p = (1-qwp)*(un1-us1)/(2*dy) + qwp*((-1*un2+4*un1-3*uvals[i,j,k])/(2*dy))

				r_m1 = (1-rwm)*(ut1-ub1)/(2*dz) + rwm*((3*uvals[i,j,k]-4*ub1+ub2)/(2*dz))
				r_p = (1-rwp)*(ut1-ub1)/(2*dz) + rwp*((-1*ut2+4*ut1-3*uvals[i,j,k])/(2*dz))

				pp = (p_m+p_p)/2.0
				qp = (q_m+q_p)/2.0
				rp = (r_m1+r_p)/2.0

				pm = (p_p-p_m)/2.0
				qm = (q_p-q_m)/2.0
				r_m = (r_p-r_m1)/2.0


				uo=c*(rhs_val-H(i,j,k,pp,qp,rp)+σx*pm+σy*qm+σz*r_m)+uvals[i,j,k]

				curr_error = abs(uo - uvals[i,j,k])
				uvals[i,j,k]=uo
				max_error = max(curr_error, max_error)

			end
		end
	end
	return max_error

end
function w∞(r::Float64, δ=0.0)
	r = (r+1.0)/2.0
    return exp(2.0/(-1.0 + (2.0*r-1.0)^2))*(145.7876577089403 -261.5195892865372*r)

end

function K∞(r::Float64)
    a=7.513931532835806

    return a*exp(2.0/(r^2-1.0))
end


function l1_error(uvals::Array{Float64},
				  U::Array{Float64},
				  V::Array{Float64},
				  W::Array{Float64},
				  edge::Array{Bool},
				  status::Array{Bool},
				  dx,
				  dy,
				  dz,
				  ϵ,
				  radius,
				  target_normal)
	 avgError=0.0
	 L1err=0
	 num_pts = 0.0
	 for I in CartesianIndices(edge)
		i = I[1]
		j = I[2]
		k = I[3]
		if status[i,j,k]# && !edge[i,j,k]
			CPx=U[i,j,k]
			CPy=V[i,j,k]
			CPz=W[i,j,k]
			gridpoint=[(i-1)*dx-1,(j-1)*dy-1,(k-1)*dz-1]
			r=norm(gridpoint)
			η=-1*(r-radius)

			σ1=1+η/r
			σ2=1+η/r
			Jac=σ1*σ2
			if abs(abs(η/ϵ) -1.0)<1e-13
				continue
			else
				kern=K∞(η/ϵ)/ϵ#(1+cos(pi*η/eps))/(2*eps)#w∞(η/eps)/eps#
			end

			uexact=exact_dist(CPx,CPy,CPz,target_normal)
			# if abs(uexact - pi*radius)< .2
			# 	continue
			# end
			uo=uvals[i,j,k]
			diff=abs(uo-uexact);

			num_pts = num_pts + 1

			L1err+=diff*dx*dy*dz*kern*Jac
			avgError+=diff
		end

  	end
	println("num pts " , num_pts)
	avgError = avgError/num_pts
	return  L1err, num_pts, avgError
end

function linf_error(uvals::Array{Float64},
					U::Array{Float64},
					V::Array{Float64},
					W::Array{Float64},
					edge::Array{Bool},
					status::Array{Bool},
					radius,
					target_normal)
	 maxdiff =0

	 ijk=[0,0,0]
	 for I in CartesianIndices(uvals)
		i = I[1]
		j = I[2]
		k = I[3]
	    if status[i,j,k]#0<uvals[i,j,k]<1000 #&& !edge[i,j,k]
			CPx=U[i,j,k]
			CPy=V[i,j,k]
			CPz=W[i,j,k]
			uexact=exact_dist(CPx,CPy,CPz,target_normal)
			uo=uvals[i,j,k]
			diff=abs(uo-uexact);
			# if abs(uexact - pi*radius)< .15
			# 	continue
			# end
			if diff>maxdiff
				maxdiff=diff
				ijk=[i,j,k]
			end
		end
  	end
	return maxdiff, ijk
end

end
