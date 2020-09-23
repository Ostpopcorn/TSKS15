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
s1 = s1/sqrt(E1);
s2 = s2/sqrt(E2);

% SNR
SNR = 30;
sigma2 = 10^(-SNR/10);

% Random time diff calc
T = -5 + 10*rand(1);

%% a)
% s2 is better because of greater derivate which is looked for in CRB
% See (3.14) in kay I
figure(1)
plot (T_range,s1,'r')
hold on
plot (T_range,s2,'b')
hold off
legend("S1","S2")

%% b)


%% Lägger till brus
x1 = exp(-0.1*(T_range-T).^2) + sqrt(sigma2)*randn(1,length(T_range));
x2 = exp(-0.1*(T_range-T).^2).*cos(T_range-T) + sqrt(sigma2)*randn(1,length(T_range));
figure(2)
plot(T_range,x1,'m')
hold on
plot(T_range,x2,'c')

%%

T_search = -5:T_s:5;
T_hat_metric = zeros(1,length(T_search));

for i = 1:1:length(T_search)
    t = T_search(i);
    intermidiate = 1/sqrt(E1)*exp(-0.1*(T_range-t).^2);
    T_hat_metric(i) = sum(x1.*intermidiate);
end

figure(3)
plot(T_search,T_hat_metric,'r')
[max, max_t_index] = max(T_hat_metric);
T_search(max_t_index)




