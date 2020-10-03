clear all
close all

T_s = 0.1;
T_range = -15:T_s:15;
N = length(T_range);
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
SNR = 5;
sigma2 = 10^(-SNR/10);

% Random time diff calc
T =-5 + 10*rand(1);
% T =2;

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
w = exp(-0.1*(T_range-T).^2) + sqrt(sigma2)*randn(1,length(T_range));
x2 = exp(-0.1*(T_range-T).^2).*cos(T_range-T) + sqrt(sigma2)*randn(1,length(T_range));

w = w/sqrt(E1);
x2 = x2/sqrt(E2);
figure(1)
plot(T_range,w,'m')
plot(T_range,x2,'c')
hold off

T_search = -5:T_s:5;
T_hat_metric = zeros(1,length(T_search));

% ML estimatorn för s1
for i = 1:1:length(T_search)
    t = T_search(i);
    intermidiate = 1/sqrt(E1)*exp(-0.1*(T_range-t).^2);
    T_hat_metric(i) = sum(w.*intermidiate);
end

genarate_T_hat_from_two_funcs(s1,w)
[max_value, max_t_index] = max(T_hat_metric);
T_search(max_t_index)
figure(3)
plot(T_search,T_hat_metric,'r')
%% Monte Carlo
% testa antalat körningar
rng('shuffle')

M0 = [1:50:1000 1001:100:2000 2001:200:5001]; % All the different number of monte-carlo runs we will try

T_hat_mean = zeros(2,M0(end));
T_hat_std = zeros(2,M0(end));

for M=M0
    
    T_hat_s1 = zeros(1,M );
    T_hat_s2 = zeros(1,M );
    for m=1:M
        % s1 and s2 contains the signal without noise.
        w =  sqrt(sigma2)*randn(1,N);
        T_hat_s1(m) = genarate_T_hat_from_two_funcs(s1,w+s1_time_diffed);
        T_hat_s2(m) = genarate_T_hat_from_two_funcs(s2,w+s2_time_diffed);
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

SNR_range = 1:4:20;
sigma2_range = 10.^(-SNR_range./10);

T_hat_std = zeros(2,length(SNR_range));
monte_carlo_runs = 2000;

for SNR_i=1:1:length(SNR_range)
    SNR_c = SNR_range(SNR_i);
    sigma2 = 10^(-SNR_c/10);

    T_hat_s1 = zeros(1,monte_carlo_runs );
    T_hat_s2 = zeros(1,monte_carlo_runs );
    for m=1:monte_carlo_runs
        % s1 and s2 contains the signal without noise.
        w =  sqrt(sigma2)*randn(1,N);
        T_hat_s1(m) = genarate_T_hat_from_two_funcs(s1,w+s1_time_diffed);
        T_hat_s2(m) = genarate_T_hat_from_two_funcs(s2,w+s2_time_diffed);
    end
    T_hat_std(1,SNR_i) = std(T_hat_s1);
    T_hat_std(2,SNR_i) = std(T_hat_s2);
    
end
std_plot_1 = T_hat_std(1,:);
std_plot_2 = T_hat_std(2,:);

figure(30)
d_s1_energy = sum(diff(s1).^2);     
semilogy(SNR_range, sqrt(sigma2_range)./d_s1_energy)
hold on 
semilogy(SNR_range, std_plot_1);
hold off

figure(31)
d_s2_energy = sum(diff(s2).^2);
semilogy(SNR_range, sqrt(sigma2_range)./d_s2_energy)
hold on 
semilogy(SNR_range, std_plot_2);
hold off
