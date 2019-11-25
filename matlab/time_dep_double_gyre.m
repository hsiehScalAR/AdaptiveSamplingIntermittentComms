function out = time_dep_double_gyre(t,X)

out = zeros(2,1);
x = X(1);
y = X(2);

I = 0.01; % noise itensity
sigma = sqrt(2*I);

%stream0 = RandStream('mt19937ar','Seed',0);         % vector field noise initialization
%RandStream.setDefaultStream(stream0);               % vector field noise initialization

eta = zeros(1,2);
eta(1) = sigma*randn();
eta(2) = sigma*randn();

A = 0.2;
mu = 0.005;
eps = 0.1;
omega = pi/3;
psi = 0;
scale = 1;

f = eps*sin(omega*t+psi).*(x.^2) + (1-2*eps*sin(omega.*t+psi)).*x;
df = 2*eps*sin(omega*t+psi).*x + (1-2*eps*sin(omega.*t+psi));

velX = -pi.*A.*sin(pi.*f./scale).*cos(pi.*y./scale) - mu*x + eta(1);
velY = pi.*A.*cos(pi.*f./scale).*sin(pi.*y./scale).*df - mu*y + eta(2);

out(1) = velX;
out(2) = velY;