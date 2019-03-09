module ErrorComp
using Interpolations
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



function exactdist(x,y,z,nT) #exact distance function on the sphere

  r=sqrt((x-.5).^2+(y-.5).^2+(z-.5).^2);
  Nxyz1=(x-.5)./r;
  Nxyz2=(y-.5)./r;
  Nxyz3=(z-.5)./r;
  Nxyz=[Nxyz1,Nxyz2,Nxyz3]
  #N=[-1,0,0]
  N=nT;
  aa = cross(Nxyz, N);
  bb = dot(Nxyz',N');
  aa=norm(aa)
  d1 = r*atan(aa, bb)


end

function maxerror(N, Uvals::Array{Float64}, U::Array{Float64}, V::Array{Float64}, W::Array{Float64},nT)#,
									#initial::Array{Bool})
	 h=1/(N-1)

	 maxdiff=0
	 ijk=[0,0,0]
	 errormat=zeros(N,N,N)
	 #exact=1000*ones(N,N,N)
	 for k= 2:N-1,
 			 j= 2:N-1,
 			 i= 2:N-1

    if 0<Uvals[i,j,k]<1000 #&& initial[i,j,k]==false
      CPx=U[i,j,k]
      CPy=V[i,j,k]
      CPz=W[i,j,k]
#uncomment if want to use tri linear interpolation to estimate Linf error
#=			iminus=(floor(Int,(CPx)/h))+1
			jminus=(floor(Int,(CPy)/h))+1
			kminus=(floor(Int,(CPz)/h))+1
			iplus=iminus+1
			jplus=jminus+1
			kplus=kminus+1
			x1=(iminus-1)*h
			y1=(jminus-1)*h
			z1=(kminus-1)*h
			x2=(iplus-1)*h
			y2=(jplus-1)*h
			z2=(kplus-1)*h

			q000=Uvals[iminus,jminus,kminus]
			q001=Uvals[iminus,jminus,kplus]
			q010=Uvals[iminus,jplus,kminus]
			q011=Uvals[iminus,jplus,kplus]
			q100=Uvals[iplus,jminus,kminus]
			q101=Uvals[iplus,jminus,kplus]
			q110=Uvals[iplus,jplus,kminus]
			q111=Uvals[iplus,jplus,kplus]


			uo=triLinterp(CPx,  CPy,  CPz,  q000,  q001, q010,  q011,  q100,  q101,  q110,
				   q111,  x1,  x2,  y1,  y2,  z1,  z2)=#

				#=	 ibottom=(floor(Int,(CPx)/h))+1-1
		 			jbottom=(floor(Int,(CPy)/h))+1-1
		 			kbottom=(floor(Int,(CPz)/h))+1-1
		 			xbottom=(ibottom-1)*h
		 			ybottom=(jbottom-1)*h
		 			zbottom=(kbottom-1)*h

		 			A_i = ibottom:(ibottom+3)
		 			A_j = jbottom:(jbottom+3)
		 			A_k = kbottom:(kbottom+3)
		 			A_x1 = xbottom:h:(xbottom+3*h)
		 			A_x2 = ybottom:h:(ybottom+3*h)
		 			A_x3 = zbottom:h:(zbottom+3*h)

		 			g=(i,j,k)->Uvals[i,j,k]
		 			Ainterp = [g(i,j,k) for i in A_i, j in A_j, k in A_k]
		 			itp = interpolate(Ainterp, BSpline(Cubic(InPlace(OnGrid()))))
		 			sitp = scale(itp, A_x1, A_x2, A_x3)

		 			uo=sitp(CPx,CPy,CPz)=#

			#println("uo= ",uo, " ")

			uexact=exactdist(CPx,CPy,CPz,nT)

			uo=Uvals[i,j,k]
			diff=abs(uo-uexact);
			errormat[i,j,k]=diff;

			if diff>maxdiff
				maxdiff=diff
				ijk=[i,j,k]
			end
		end

  end

	return maxdiff
end

function L1error(N, Uvals::Array{Float64}, U::Array{Float64}, V::Array{Float64}, W::Array{Float64},ϵ,radius,nT)
									#initial::Array{Bool},
	 h=1/(N-1)
	# errormat=zeros(N,N,N)
	 L1err=0
	 numpts=0
	 for k= 2:N-1,
 			 j= 2:N-1,
 			 i= 2:N-1

    if 0<Uvals[i,j,k]<500 #&& initial[i,j,k]==false
      CPx=U[i,j,k]
      CPy=V[i,j,k]
      CPz=W[i,j,k]
			gridpoint=[(i-1)*h,(j-1)*h,(k-1)*h]
			r=norm(gridpoint-[.5,.5,.5])
			η=-1*(r-radius)
			σ1=1+η/r
			σ2=1+η/r
			Jac=σ1*σ2
			kern=(1+cos(pi*η/ϵ))/(2*ϵ)

			uexact=exactdist(CPx,CPy,CPz,nT)
			uo=Uvals[i,j,k]
			diff=abs(uo-uexact);
			L1err+=diff*h*h*h*kern*Jac




		end

  end

	return  L1err
end

function maxerrorInterp(N, Uvals::Array{Float64}, U::Array{Float64}, V::Array{Float64}, W::Array{Float64},nT)#,
									#initial::Array{Bool})
	 h=1/(N-1)

	 maxdiff=0
	 ijk=[0,0,0]
	 errormat=zeros(N,N,N)

	 for k= 2:N-1,
 			 j= 2:N-1,
 			 i= 2:N-1

    if 0<Uvals[i,j,k]<1000 #&& initial[i,j,k]==false
      CPx=U[i,j,k]
      CPy=V[i,j,k]
      CPz=W[i,j,k]

			iminus=(floor(Int,(CPx)/h))+1
			jminus=(floor(Int,(CPy)/h))+1
			kminus=(floor(Int,(CPz)/h))+1
			iplus=iminus+1
			jplus=jminus+1
			kplus=kminus+1
			x1=(iminus-1)*h
			y1=(jminus-1)*h
			z1=(kminus-1)*h
			x2=(iplus-1)*h
			y2=(jplus-1)*h
			z2=(kplus-1)*h

			q000=Uvals[iminus,jminus,kminus]
			q001=Uvals[iminus,jminus,kplus]
			q010=Uvals[iminus,jplus,kminus]
			q011=Uvals[iminus,jplus,kplus]
			q100=Uvals[iplus,jminus,kminus]
			q101=Uvals[iplus,jminus,kplus]
			q110=Uvals[iplus,jplus,kminus]
			q111=Uvals[iplus,jplus,kplus]


			uo=triLinterp(CPx,  CPy,  CPz,  q000,  q001, q010,  q011,  q100,  q101,  q110,
				   q111,  x1,  x2,  y1,  y2,  z1,  z2)



			uexact=exactdist(CPx,CPy,CPz,nT)

			diff=abs(uo-uexact);
			errormat[i,j,k]=diff;

			if diff>maxdiff
				maxdiff=diff

			end
		end

  end

	return maxdiff
end

end
