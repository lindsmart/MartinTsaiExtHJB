
function main(N, center_pnt, niters)
	h = 2.0/(N-1)

	i0 = Int64(1+(center_pnt[1]+1)/h)
	j0 = Int64(1+(center_pnt[2]+1)/h)

	uvals = 1000*ones(N,N)
	rhs = ones(N,N)
	uvals[i0,j0] = 0.0
	H(p,q) = sqrt(p^2+q^2)

	σx = 1.0
	σy = 1.0
	pp,qp=0.0, 0.0
	pm,qm=0.0, 0.0
	rhs_val = 0.0
	uw, ue, uo = 0.0, 0.0, 0.0, 0.0, 0.0
	c= 0.0
	nx, ny = map(dim->size(uvals, dim), (1,2))
	for n=1:niters
	   	for sgn1 in (-1,1), sgn2 in (-1,1)
			for j= (sgn2<0 ? ny : 1): sgn2 : (sgn2<0 ? 1 : ny),
				i= (sgn1<0 ? nx : 1): sgn1 : (sgn1<0 ? 1 : nx)
				if max(i,j)==N || min(i,j)==1
					if i==1
						u0=max(2*uvals[i+1,j]-uvals[i+2,j],uvals[i+2,j])
						uvals[i,j]=min(u0,uvals[i,j])
					end
					if i==nx
						u0=max(2*uvals[i-1,j]-uvals[i-2,j],uvals[i-2,j])
						uvals[i,j]=min(u0,uvals[i,j])
					end
					if j==1
						u0=max(2*uvals[i,j+1]-uvals[i,j+2],uvals[i,j+2])
						uvals[i,j]=min(u0,uvals[i,j])
					end
					if j==ny
						u0=max(2*uvals[i,j-1]-uvals[i,j-2],uvals[i,j-2])
						uvals[i,j]=min(u0,uvals[i,j])
					end

				else

					c=1/(σx/h+σy/h)
					rhs_val=rhs[i,j]

					uw=uvals[i-1,j];ue=uvals[i+1,j];
					us=uvals[i,j-1];un=uvals[i,j+1];

					pp=(ue+uw)/(2*h)
					qp=(un+us)/(2*h)
					pm=(ue-uw)/(2*h)
					qm=(un-us)/(2*h)
					uo=c*(rhs_val-H(pm,qm)+σx*pp+σy*qp)
					if uo<uvals[i,j]
						uvals[i,j]=uo
					end
				end
			end
		end
	end




	exact = [sqrt(((i-1)*h-1)^2+((j-1)*h-1)^2) for i in range(1,N, length=N), j in range(1,N,length=N)]

	diff = broadcast(abs, (exact-uvals))
	return maximum(diff)
end

center_pnt = [0.,0.]
niters = 300
Linfs=ones(6)
orders = ones(5)
sig = 1.
gridsize = [51, 101, 201, 401, 801]
for index = 1:5
    N = gridsize[index]
    Linf = main(N,center_pnt,niters)
    Linfs[index]=Linf
    if index>1
        order = log(Linfs[index]/Linfs[index-1])/log(2.0/gridsize[index]/(2.0/gridsize[index-1]))
        print("\n" ,order)
        orders[index-1]=order
    end
end
