clear all
close all

T_s = 0.1;
T_range = [-15:T_s:15];
s1 = exp(-0.1*T_range.^2);
s2 = exp(-0.1*T_range.^2).*cos(T_range);

% Compute the energy
E1 = sum(abs(s1.^2));
E2 = sum(abs(s2.^2));

% Normalized signals => signal energy =1
alpha_1 = 1/sqrt(E1);
alpha_2 = 1/sqrt(E2);
s1 = s1*alpha_1;
s2 = s2*alpha_2;

% SNR
SNR = 25;
sigma2 = 10^(-SNR/10);

% Random time diff calc
T = 2% -5 + 10*rand(1);

s1_time_diffed = exp(-0.1*(T_range-T).^2);
s2_time_diffed = exp(-0.1*(T_range-T).^2).*cos((T_range-T));

%% a)
% s2 is better because of greater derivate which is looked for in CRB See
% (3.14) in kay I
figure(1)
plot (T_range,s1,'r')
hold on
plot (T_range,s2,'b')
legend("S1","S2")

% Lägger till brus
x1 = exp(-0.1*(T_range-T).^2) + sqrt(sigma2)*randn(1,length(T_range));
x2 = exp(-0.1*(T_range-T).^2).*cos(T_range-T) + sqrt(sigma2)*randn(1,length(T_range));

x1 = x1/sqrt(E1);
x2 = x2/sqrt(E2);
figure(1)
plot(T_range,x1,'m')
plot(T_range,x2,'c')
hold off

T_search = -5:T_s:5;
T_hat_metric = zeros(1,length(T_search));

% ML estimatorn för s1
for i = 1:1:length(T_search)
    t = T_search(i);
    intermidiate = 1/sqrt(E1)*exp(-0.1*(T_range-t).^2);
    T_hat_metric(i) = sum(x1.*intermidiate);
end

[max, max_t_index] = max(T_hat_metric);
conv_val = genarate_T_hat_from_two_funcs(s1,x1)
search_val = T_search(max_t_index)
figure(3)
plot(T_search,T_hat_metric,'r')
%% Monte Carlo
% testa antalat körningar
rng('shuffle')

M0 = 1:100:5001; % All the different number of monte-carlo runs we will try
N = length(T_range);
T_hat_mean = zeros(2,M0(end));
T_hat_std = zeros(2,M0(end));

for M=M0
    
    T_hat_s1 = zeros(1,M );
    T_hat_s2 = zeros(1,M );
    for m=1:M
        % s1 and s2 contains the signal without noise.
        w1 =  sqrt(sigma2)*rand(1,N);
        T_hat_s1(m) = genarate_T_hat_from_two_funcs(s1,w1+s1_time_diffed);
        T_hat_s2(m) = genarate_T_hat_from_two_funcs(s2,w1+s2_time_diffed);
    end
    T_hat_mean(1,M) = mean(T_hat_s1);
    T_hat_std(1,M) = std(T_hat_s1);
    T_hat_mean(2,M) = mean(T_hat_s2);
    T_hat_std(2,M) = std(T_hat_s2);
    
end

mean_plot_1 = T_hat_mean(1,:);
std_plot_1 = T_hat_std(1,:);
mean_plot_2 = T_hat_mean(2,:);
std_plot_2 = T_hat_std(2,:);

figure(25)
plot(M0,mean_plot_1(M0),M0,mean_plot_2(M0))
legend("s1","s2")
title("mean")

figure(26)
plot(M0,std_plot_1(M0),M0,std_plot_2(M0))
legend("s1","s2")
title("std")

%% CRB

%Plot CRB theoretical value

SNR_range = 1:1:40;
sigma2_range = 10.^(-SNR_range./10);

d_s1_energy = sum(diff(s1).^2);
semilogy(SNR_range, sigma2_range/d_s1_energy)
figure(30)
d_s1_energy = sum(diff(s1).^2);
semilogy(SNR_range, sigma2_range/d_s1_energy)
% semilogy(SNR_range,sigma2_range(SNR_range),'b-')

figure(31)
d_s2_energy = sum(diff(s2).^2);
semilogy(SNR_range, sigma2_range/d_s2_energy)

