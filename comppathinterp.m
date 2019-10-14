u4=(Uvals);
N=size(Uvals,1);
% X0=x0y0z03.Position(1);
% Y0=x0y0z03.Position(2);
% Z0=x0y0z03.Position(3);
X0=TwoPtsv1.Position(1);
Y0=TwoPtsv1.Position(2);
Z0=TwoPtsv1.Position(3);

Target=T2Bunny;

TargetX = round((N-1)*(Target(1))+1); 
TargetY = round((N-1)*(Target(2))+1);
TargetZ = round((N-1)*(Target(3))+1);

% arc length 
arcLength = 0.5;

% tolerance
TOL = arcLength*3;

% start path vector
xv = X0;
yv = Y0;
zv = Z0;
i=0;

% loop while goal is not within a tolerance
while (sqrt( (xv(end)-Target(1))^2 + (yv(end)-Target(2))^2 +(zv(end)-Target(3))^2 ) > .03)
    
    % STEP 1: compute gradient by nearest neighbor approximation
     xn = floor(xv(end)*(N-1)+1);
    yn = floor(yv(end)*(N-1)+1);
    zn = floor(zv(end)*(N-1)+1);
   
    U1=ones(2,2,2);
     U2=ones(2,2,2);
      U3=ones(2,2,2);
    for l=xn:(xn+1)
        for t=yn:(yn+1)
            for o=zn:(zn+1)
                U1(rem(l,xn)+1,rem(t,yn)+1,rem(o,zn)+1)=Ux(l,t,o);
               U2(rem(l,xn)+1,rem(t,yn)+1,rem(o,zn)+1)=Uy(l,t,o);
                U3(rem(l,xn)+1,rem(t,yn)+1,rem(o,zn)+1)=Uz(l,t,o);
            end
        end
    end
     xnp = xn + 1;
    xnp=(xnp-1)/(N-1);
    %xnn = xn - 1;
    ynp = yn + 1;
    ynp=(ynp-1)/(N-1);
    %ynn = yn - 1;
    znp = zn + 1;
    znp=(znp-1)/(N-1);
    %znn = zn - 1;
    
     xn=(xn-1)/(N-1);
     yn=(yn-1)/(N-1);
     zn=(zn-1)/(N-1);
    % check if its too close to the edge
   
      [X,Y,Z]=meshgrid(xn:1/(N-1):xnp,yn:1/(N-1):ynp,zn:1/(N-1):znp);

    xq=interp3(X,Y,Z,U1,xv(end),yv(end),zv(end),'makima');
    yq=interp3(X,Y,Z,U2,xv(end),yv(end),zv(end),'makima');
    zq=interp3(X,Y,Z,U3,xv(end),yv(end),zv(end),'makima');
   % grad_norm = sqrt(gradx^2 + grady^2+gradz^2);
    
    if (grad_norm == 0)
        disp('ERROR: Initial point is not reachable.'); 
        xv = -100;
        yv = -100;
        zv = -100;
        break;
    end
    
    % STEP 2: advance point in space (x,y)
     xv = [xv, xv(end)+xq*arcLength];  %#ok<AGROW>
    yv = [yv, yv(end)+yq*arcLength];%#ok<AGROW>
    zv = [zv, zv(end)+zq*arcLength];%#ok<AGROW>
    
   
    
    
end

%scale back down to [-L,L] space:
xvs = (xv-1)/(N-1);
yvs = (yv-1)/(N-1);
zvs= (zv-1)/(N-1);


% figure;
% pic = patch(isosurface(x, y, z, smooth3(compdist),0, Uvals));
%   
%   pic.FaceColor = 'interp';
% pic.EdgeColor = 'none';
% 
%  set(gca,'visible','off')
% daspect([1 1 1]); axis tight;
% view([0,90])
% colormap(jet)
% camlight('right'); lighting flat
% material dull
hold on
plot3(xvs,yvs,zvs,'r-','LineWidth',3);



ylim([0 1])
xlim([0 1])
zlim([0,1])

%Uncomment below to plot without contour of value function
% figure(99),clf;
% contour(x,y,R,[0 0],'b-','LineWidth',4); hold on;
% plot(TargetX,TargetY,'k+','LineWidth',4);
% plot(X0,X0,'mo','LineWidth',4);
% contour(x,y, u2, [B B],'k-','LineWidth',4);
% plot(xvs,yvs,'r.-');
% axis square;
% legend('region bdry','goal','start','feasible bdry','optimal path','Location','BestOutside');


