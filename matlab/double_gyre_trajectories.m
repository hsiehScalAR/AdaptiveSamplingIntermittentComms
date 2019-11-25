%double gyre flow trajectories

clf
hold on
axis([0 2 0 1]);

xo = [.999 .2];
tspan = [0 1];
[t y] = ode45(@double_gyre_func,tspan,xo);
plot(y(:,1),y(:,2))

xo = [.998 .2];
tspan = [0 1];
[t y] = ode45(@double_gyre_func,tspan,xo);
plot(y(:,1),y(:,2))

xo = [1.001 .2];
tspan = [0 1];
[t y] = ode45(@double_gyre_func,tspan,xo);
plot(y(:,1),y(:,2))



x = linspace(0,2,40);
y = linspace(0,2,40);

[X Y] = meshgrid(x,y);

u = -pi().*sin(pi().*X).*cos(pi().*Y);
v = pi().*sin(pi().*Y).*cos(pi().*X);

quiver(X,Y,u,v);

hold off