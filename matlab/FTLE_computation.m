% FTLE_computation.m
%
% Computes the FTLE field at each time step. This script is intended to be
% used with an underlying velocity function (such as the double gyre or
% time dependent double gyre)
% 
% Author: Matt Michini (2012)
% Drexel University Scalable Autonomous Systems lab
% All rights reserved

clear

T = 2.0; % integration time in sec, 4.5 for potentialvortices2

 xmin = -2.8;
 xmax = 2.8;
 ymin = -2.8;
 ymax = 0.8;
 tmin = 0;
 tmax = 20 + T;

 xmin = -0.1;
 xmax = 2.1;
 ymin = -0.1;
 ymax = 2.1;
 
gridsize = 600; % This is the size of the initial grid of particles for the FTLE computation

x0 = linspace(xmin,xmax,gridsize);
y0 = linspace(ymin,ymax,gridsize);

[X0 Y0] = ndgrid(x0,y0); % initial particle grid

X_adv = zeros(gridsize);
Y_adv = zeros(gridsize); % These matrices store the x,y coordinates of the
% advected point originating at the corresponding X0,Y0 value.

k = 1;
clear mov;
% set(gcf,'Position',[0 0 600 600]);

dt = .05; % 'frame rate' for new data. original dt is 1/7.5. 1/30 will produce a 30 FPS video
timespan = 0:dt:3.5;

%FTLE_array = zeros(gridsize,gridsize,length(timespan)); % this array stores all of the FTLE information for the entire run

for t0 = timespan(1)
    for i = 1:gridsize
        tic
        for j = 1:gridsize
            [t Y_out] = ode45(@double_gyre_func,[t0 t0+T/2 t0+T],[X0(i,j) Y0(i,j)]); % odefunc must return the velocity
            X_adv(i,j) = Y_out(3,1); % take the final x position
            Y_adv(i,j) = Y_out(3,2); % take the final y position
        end
        disp(i);
    end
    
    % now do finite differencing to get the gradient of theflow map at 
    % each point. Notice that we start at 2 and end at gridsize-1, since we 
    % are doing central differencing. After getting the 2x2 gradient, we
    % find the largest eigenvalue of its square, then plug it into the
    % FTLE eqn.
    
    phi = zeros(2);
    FTLE = zeros(gridsize);
    
    for i = 2:(gridsize-1)
        for j = 2:(gridsize-1)
            phi(1,1) = (X_adv(i+1,j)-X_adv(i-1,j))/(X0(i+1,j)-X0(i-1,j));
            phi(1,2) = (X_adv(i,j+1)-X_adv(i,j-1))/(Y0(i,j+1)-Y0(i,j-1));
            phi(2,1) = (Y_adv(i+1,j)-Y_adv(i-1,j))/(X0(i+1,j)-X0(i-1,j));
            phi(2,2) = (Y_adv(i,j+1)-Y_adv(i,j-1))/(Y0(i,j+1)-Y0(i,j-1));

            delta = phi'*phi; % compute rotation-independent deformation tensor 
            lambda_max = max(eig(delta)); % Find the maximum eigenvalue  
            FTLE(i,j)= 1/abs(T)*log(sqrt(lambda_max)); % the FTLE formula
        end
    end
    
    %FTLE_array(:,:,k) = FTLE;
    
    clf
    [~,ch] = contourf(X0,Y0,FTLE,20); % the last argument is the number of contour levels. A higher value will take longer.
    set(ch,'edgecolor','none'); % remove the lines between contour levels
    daspect([1 1 1]); % set the aspect ratio to 1:1
    
    %caxis([0 2.5]); % you may need to adjust this for display purposes
    
    %fileName = sprintf('FTLE_frame%d.fig', k);
    %saveas(gcf,fileName,'fig');
    
    % Capture the movie frame and increment the movie counter.
    mov(k) = getframe;
    k = k+1;
end

% Write the video:
% obj = VideoWriter('time_dep_sinusoid_oscillation_500x500_T2_5.avi');
% obj.FrameRate = (1/dt);
% open(obj);
% writeVideo(obj,mov);
% close(obj);

