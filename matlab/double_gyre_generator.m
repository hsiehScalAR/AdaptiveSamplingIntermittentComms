clear all

gridsize = 55;

[x y] = meshgrid(linspace(-2,4,gridsize),linspace(-2,4,gridsize));

u = zeros(gridsize);
v = zeros(gridsize);

nt = 5;
tmin = 0;
tmax = 12;
t_vect = linspace(tmin,tmax,nt);

for k = 1:nt
    for i = 1:gridsize
        for j = 1:gridsize
%             [u_out] = time_dep_double_gyre(t_vect(k),[x(i,j) , y(i,j)]); 
%             u(i,j) = u_out(1);
%             v(i,j) = u_out(2);
            u(i,j) = -pi*sin(pi*x(i,j))*cos(pi*y(i,j));
            v(i,j) = pi*cos(pi*x(i,j))*sin(pi*y(i,j));
        end
    end
    U(:,:,k) = u;
    V(:,:,k) = v;
    quiver(x,y,u,v);
    pause(0.1);
    disp(k);
end

%U = repmat(u,[1 1 nt]);
%V = repmat(v,[1 1 nt]);

%save('double_gyre','x','y','t_vect','U','V');