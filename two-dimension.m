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
x_init = -2:0.8:2; y_init = -2:0.8:2;
% x_init(6) = [];
% y_init(6) = [];

%% for kernel
% x_init = 1; y_init = 1;

%%
[x_init_grid, y_init_grid] = meshgrid(x_init,y_init);
data_init = [reshape(y_init_grid,1,length(y_init)^2);reshape(x_init_grid,1,length(x_init)^2)];

for count = 1:length(x_init)^2
    x0 = data_init(1,count);
    y0 = data_init(2,count);
    X0 = ones(n_samples,2);
    X0(:,1) = x0*X0(:,1);
    X0(:,2) = y0*X0(:,2);
    x = zeros(Nt, n_samples);
    y = zeros(Nt, n_samples);
    x(1,:) = X0(:,1)';
    y(1,:) = X0(:,2)';
    for i = 1:Nt-1
        M=stblrnd(alpha/2,1,2*(dt*cos(pi*alpha/4))^(2/alpha),0,1,n_samples);
        Normal=randn(2,n_samples);
%         Bh=sqrt(dt)*randn(2,n_samples);
        Ut = sqrt(M).*Normal(1,:);
        Vt = sqrt(M).*Normal(2,:);
%         x(i+1,:) = x(i,:) + (3*x(i,:) - 1*x(i,:).^3)*dt + 1*Ut + 0*(1*x(i,:)+0).*Bh(1,:);
%         y(i+1,:) = y(i,:) + (3*y(i,:) - 1*y(i,:).^3)*dt + 1*Vt + 0*(1*y(i,:)+0).*Bh(2,:);
%         x(i+1,:) = x(i,:) + (1*x(i,:) - 1*x(i,:).^3 - 1*x(i,:).*y(i,:).^2)*dt + 1*Ut + 0*cos(1*x(i,:)+0).*Bh(1,:);
%         y(i+1,:) = y(i,:) - ((1 + x(i,:).^2).*y(i,:))*dt + 1*Vt + 0*(1*y(i,:)+0).*Bh(2,:);
%         x(i+1,:) = x(i,:) + (0.001*x(i,:) - x(i,:).*y(i,:))*dt + 1*Ut + (1*x(i,:)+0).*Bh(1,:);
%         y(i+1,:) = y(i,:) + (-6*y(i,:) + 0.25*x(i,:).^2)*dt + 1*Vt + 1*(1*y(i,:)+0).*Bh(2,:);
        x(i+1,:) = x(i,:) + (5*x(i,:) - y(i,:).^2)*dt + 1*Ut;
        y(i+1,:) = y(i,:) + (5*x(i,:) + y(i,:))*dt + 1*Vt;
    %     zxf=(z0x-z0x.^3-5*z0x.*z0y.^2)*h+(1+z0y).*Bh(1,:)+Bh(2,:)+sigma*sqrt(M).*Normal(1,:);
    %     zxf0=z0x;
    %     zyf=-(1+z0x.^2).*z0y*h+z0x.*Bh(2,:)+sigma*sqrt(M).*Normal(2,:);
    %     zyf0=z0y;
    end
    x_end = x(end,:); y_end = y(end,:);
    path = sprintf('Data_%d.mat',count);
    save(path,'x_end','y_end')
    count
    max(x_end)
    max(y_end)
    
%     save('DW_sde.mat','x_end','y_end')
end
% plot(t,x)

%% compute alpha and sigma
R=[];
for i=1:length(x_init)^2
    path = sprintf('Data_%d.mat',i);
    load(path);
    R=[R sqrt((x_end-data_init(1,i)).^2+(y_end-data_init(2,i)).^2)];
end
d=2;
gammaEuler=-psi(1)*gamma(1);
ElogR=sum(log(R))/length(R);
varlogR=var(log(R));
alpha0=(6/pi^2*(varlogR-0.25*psi(1,d/2))+0.25)^(-0.5);
sigma0=0.5*exp(ElogR-gammaEuler*(1/alpha0-0.5)-0.5*psi(d/2))*dt^(-1/alpha0);

