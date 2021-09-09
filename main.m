clc;clear;
rand('seed',1)
randn('seed',1)
alpha = 1.5;
n_samples = 5000;
% T = 0.001;
% dt = 0.0001;
T = 0.001;
dt = 0.001;
t = 0:dt:T;
Nt = length(t);
%% for drift
x_init = linspace (-2.5,2.5,21);
% x_init(6) = [];
% y_init(6) = [];

%% for kernel
% x_init = 1; y_init = 1;

%%

for count = 1:length(x_init)
    X0=x_init(count)* ones(n_samples,1);
    x = zeros(Nt, n_samples);
    x(1,:) = X0(:,1)';
    for i = 1:Nt-1
        M=dt^(1/alpha)*stblrnd(alpha,0,1,0,1,n_samples);
        x(i+1,:) = x(i,:) + (4*x(i,:) - x(i,:).^3)*dt + M;
    end
    x_end = x(end,:);
    path = sprintf('Data_%d.mat',count);
    save(path,'x_end')
    count
    max(x_end)
    
%     save('DW_sde.mat','x_end','y_end')
end
% plot(t,x)

%% compute alpha and sigma
R=[];
for i=1:length(x_init)
    path = sprintf('Data_%d.mat',i);
    load(path);
    R=[R abs(x_end-x_init(i))];
end
d=1;
gammaEuler=-psi(1)*gamma(1);
ElogR=sum(log(R))/length(R);
varlogR=var(log(R));
alpha0=(6/pi^2*(varlogR-0.25*psi(1,d/2))+0.25)^(-0.5);
sigma0=0.5*exp(ElogR-gammaEuler*(1/alpha0-0.5)-0.5*psi(d/2))*dt^(-1/alpha0);

