load('control.mat')

u4=(Uvals);

N=size(Uvals,1);
[x,y,z]=ndgrid(0:1/(N-1):1);
alp=0.005;
plotbun=1;
markcol='k-';
 xyzani=pt1skull;

[~,indxT]=ismember(xyzani.Position,FVskull.vertices,'rows');
X0=xyzani.Position(1)+alp*normals(indxT,1);
Y0=xyzani.Position(2)+alp*normals(indxT,2);
Z0=xyzani.Position(3)+alp*normals(indxT,3);
speed=ones(N,N,N);

% control1=control1;
% control2=control2;
% control3=control3;
% % initial point in pixels
% x0 = round((N-1)*(X0)+1); 
% y0 = round((N-1)*(Y0)+1);
% z0 = round((N-1)*(Z0)+1);
[~,indx]=ismember(skullT,FVskull.vertices,'rows');
Target=skullT+alp*normals(indx,:);

TargetX = round((N-1)*(Target(1))+1); 
TargetY = round((N-1)*(Target(2))+1);
TargetZ = round((N-1)*(Target(3))+1);

% arc length 
arcLength = 0.001;

% tolerance
TOL = arcLength*3;

% start path vector
xv = X0;
yv = Y0;
zv = Z0;
i=0;
% loop while goal is not within a tolerance
while (sqrt( (xv(end)-Target(1))^2 + (yv(end)-Target(2))^2 +(zv(end)-Target(3))^2 ) > .03)
    i=i+1;
     if i>100000
%           print('error too many iter');
        break;
        
    end
   
    xn = floor(xv(end)*(N-1)+1);
    yn = floor(yv(end)*(N-1)+1);
    zn = floor(zv(end)*(N-1)+1);
   
    controlsamps1=ones(2,2,2);
     controlsamps2=ones(2,2,2);
      controlsamps3=ones(2,2,2);
    for l=xn:(xn+1)
        for t=yn:(yn+1)
            for o=zn:(zn+1)
                controlsamps1(rem(l,xn)+1,rem(t,yn)+1,rem(o,zn)+1)=control1(l,t,o);
                controlsamps2(rem(l,xn)+1,rem(t,yn)+1,rem(o,zn)+1)=control2(l,t,o);
                controlsamps3(rem(l,xn)+1,rem(t,yn)+1,rem(o,zn)+1)=control3(l,t,o);
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
%     if (xn < 2)||(xn > N-1)||(yn < 2)||(yn > N-1)||(zn < 2)||(zn > N-1)
%         disp('WARNING: Path outside domain or too close to edge'); 
%         break;
%     end
    [X,Y,Z]=meshgrid(xn:1/(N-1):xnp,yn:1/(N-1):ynp,zn:1/(N-1):znp);
    
    
    xq=interp3(X,Y,Z,controlsamps1,xv(end),yv(end),zv(end),'makima');
    yq=interp3(X,Y,Z,controlsamps2,xv(end),yv(end),zv(end),'makima');
    zq=interp3(X,Y,Z,controlsamps3,xv(end),yv(end),zv(end),'makima');
    
    % compute gradient
%     gradx = u4(xnp,yn,zn) - u4(xnn,yn,zn); 
%     grady = u4(xn,ynp, zn) - u4(xn,ynn, zn); 
%     gradz = u4(xn, yn, znp) - u4(xn, yn, znn); 
    
%     grad_norm = sqrt(gradx^2 + grady^2+gradz^2);
    
%     if (grad_norm == 0)
%         disp('ERROR: Initial point is not reachable.'); 
%         xv = -100;
%         yv = -100;
%         zv = -100;
%         break;
%     end


    
    % STEP 2: advance point in space (x,y)
    xv = [xv, xv(end)+xq*arcLength];  %#ok<AGROW>
    yv = [yv, yv(end)+yq*arcLength];%#ok<AGROW>
    zv = [zv, zv(end)+zq*arcLength];%#ok<AGROW>
    
   
    
    
end
xv=[xv,Target(1);];
yv=[yv,Target(2)];
zv=[zv,Target(3)];
xvskull1=smooth(xv,'loess');
yvskull1=smooth(yv,'loess');
zvskull1=smooth(zv,'loess');
% scale back down to [-L,L] space:
% xvs = (xv-1)/(N-1);
% yvs = (yv-1)/(N-1);
% zvs = (zv-1)/(N-1);

if plotbun
figure;
pic = patch(isosurface(x, y, z, smooth3(u),0,Uvals));
  
  pic.FaceColor = 'interp';
pic.EdgeColor = 'none';
 pic.FaceAlpha=.7;
  set(gca,'visible','off')
 direction=[1,0,0];
%   rotate3d(pic,direction,-270)
daspect([1 1 1]); axis tight;
view([-5,-8])
% gca.CameraPosition
colormap(hsv)
camlight('right'); lighting flat
material dull
end
hold on
c=[xv;yv;zv];
p=plot3(xv,yv,zv,markcol,'LineWidth',3);

%  rotate3d(p,direction,-270)
%
% 
% 
% 
% ylim([0 1])
% xlim([0 1])
% zlim([0,1])

%Uncomment below to plot without contour of value function
% figure(99),clf;
% contour(x,y,R,[0 0],'b-','LineWidth',4); hold on;
% plot(TargetX,TargetY,'k+','LineWidth',4);
% plot(X0,X0,'mo','LineWidth',4);
% contour(x,y, u2, [B B],'k-','LineWidth',4);
% plot(xvs,yvs,'r.-');
% axis square;
% legend('region bdry','goal','start','feasible bdry','optimal path','Location','BestOutside');


