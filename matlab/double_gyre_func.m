function out = double_gyre_func(~,X)

u = -pi().*sin(pi().*X(1)).*cos(pi().*X(2));
v = pi().*sin(pi().*X(2)).*cos(pi().*X(1));

out = [u;v];