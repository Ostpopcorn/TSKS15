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
sigma2 = 
SNR = -10*log10(sigma2);

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

% Lägger till brus
w = exp(-0.1*(T_range-T).^2) + sqrt(sigma2)*randn(1,length(T_range));
x2 = exp(-0.1*(T_range-T).^2).*cos(T_range-T) + sqrt(sigma2)*randn(1,length(T_range));

w = w/sqrt(E1);
x2 = x2/sqrt(E2);
figure(1)
plot(T_range,w,'m')
plot(T_range,x2,'c')
legend("s_1","s_2","x_1","x_2")
title("Comparison of original and delayed with white noise")
xlabel("T")
ylabel("Amplitude")
hold off

T_search = -5:T_s:5;
T_hat_metric = zeros(1,length(T_search));

% ML estimatorn för s1
for i = 1:1:length(T_search)
    t = T_search(i);
    intermidiate = 1/sqrt(E1)*exp(-0.1*(T_range-t).^2);
    T_hat_metric(i) = sum(w.*intermidiate);
end

conv_value = genarate_T_hat_from_two_funcs(s1,w)
[max_value, max_t_index] = max(T_hat_metric);
grid_value = T_search(max_t_index)

figure(3)
plot(T_search,T_hat_metric,'r')
title("ML estimate evaluation")
xlabel("\tau")
ylabel("Amplitude")
%% Monte Carlo , var och mean
% testa antalat körningar
rng('shuffle')

M0 = [1:50:1000 1001:100:2000 2001:100:6001]; % All the different number of monte-carlo runs we will try

T_hat_mean = zeros(2,M0(end));
T_hat_RMSE = zeros(2,M0(end));


for M=M0
    
    T_hat_error_square_s1 = zeros(1,M );
    T_hat_error_square_s2 = zeros(1,M );
    parfor m=1:M
        % s1 and s2 contains the signal without noise.
        w =  sqrt(sigma2)*randn(1,N);
        T_hat_error_square_s1(m) = genarate_T_hat_from_two_funcs(s1,w+s1_time_diffed);
        T_hat_error_square_s2(m) = genarate_T_hat_from_two_funcs(s2,w+s2_time_diffed);
    end
    T_hat_mean(1,M) = mean(T_hat_error_square_s1);
    T_hat_RMSE(1,M) = std(T_hat_error_square_s1);
    T_hat_mean(2,M) = mean(T_hat_error_square_s2);
    T_hat_RMSE(2,M) = std(T_hat_error_square_s2);
    
end

mean_plot_1 = T_hat_mean(1,:);
RMSE_plot_1 = T_hat_RMSE(1,:);
mean_plot_2 = T_hat_mean(2,:);
RMSE_plot_2 = T_hat_RMSE(2,:);

figure(25)
plot(M0,mean_plot_1(M0),'r',M0,mean_plot_2(M0),'b')
legend("\mu_{s_1}","\mu_{s_2}")
title("Monte-Carlo siumulation of \mu")

figure(26)
plot(M0,RMSE_plot_1(M0),M0,RMSE_plot_2(M0))
legend("\sigma^2_{s_1}","\sigma^2_{s_2}")
xlabel("")
title("Monte-Carlo siumulation of \sigma^2")


%% CRB
%Plot CRB theoretical value
rng('shuffle')

sigma2_range = 10^-2:10^-3:10^-1;
SNR_range = -10*log10(sigma2_range);
% SNR_range = 1:2:30;
% sigma2_range = 10.^(-SNR_range./10);

T_hat_RMSE = zeros(2,length(SNR_range));
monte_carlo_runs = 10000;

for SNR_i=1:1:length(SNR_range)
    SNR_c = SNR_range(SNR_i);
    sigma2 = sigma2_range(SNR_i);

    T_hat_error_square_s1 = zeros(1,monte_carlo_runs );
    T_hat_error_square_s2 = zeros(1,monte_carlo_runs );
    parfor m=1:monte_carlo_runs
        % s1 and s2 contains the signal without noise.
        T =-5 + 10*rand(1);
        s1_time_diffed = alpha_1*exp(-0.1*(T_range-T).^2);
        s2_time_diffed = alpha_2*exp(-0.1*(T_range-T).^2).*cos((T_range-T));
        w =  sqrt(sigma2)*randn(1,N);
        T_hat_error_square_s1(m) = (T-genarate_T_hat_from_two_funcs(s1,w+s1_time_diffed))^2;
        T_hat_error_square_s2(m) = (T-genarate_T_hat_from_two_funcs(s2,w+s2_time_diffed))^2;
    end
    T_hat_RMSE(1,SNR_i) = sqrt(mean(T_hat_error_square_s1));
    T_hat_RMSE(2,SNR_i) = sqrt(mean(T_hat_error_square_s2));
    
end
RMSE_plot_1 = T_hat_RMSE(1,:);
RMSE_plot_2 = T_hat_RMSE(2,:);
%%
% finns nog ett fel i mina teoretiska värden

s1_diff = alpha_1*exp(-0.1*T_range.^2)*(-0.2).*T_range;
s2_diff = alpha_2*exp(-0.1*T_range.^2).*(-0.2.*T_range.*cos(T_range)- sin(T_range));

figure(30)  
semilogy(SNR_range, RMSE_plot_1); 
hold on 
d_s1_energy = sum(s1_diff.^2);  
semilogy(SNR_range, sqrt(sigma2_range./d_s1_energy))

semilogy(SNR_range, RMSE_plot_2);
d_s2_energy = sum(s2_diff.^2);
semilogy(SNR_range, sqrt(sigma2_range./d_s2_energy))
title("RSME error and \surd{}CRB for s_1 and s_2")
ylabel("RMSE")
xlabel("SNR [dB]")
hold off
legend("RMSE(s_1)","\surd{}CRB(s_1)","RMSE(s_2)","\surd{}CRB(s_2)")





