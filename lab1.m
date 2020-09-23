clear all
close all

Ts = 0.1;
T_range = [-15:Ts:15];
s1 = exp(-0.1*T_range.^2);
s2 = exp(-0.1*T_range.^2).*cos(T_range);
figure(1)
plot (T_range,s1)
hold on
plot (T_range,s2)
hold off
legend("S1","S2")

E1 = sum(abs(s1.^2));
E2 = sum(abs(s2.^2));
s1 = s1/sqrt(E1);
s2 = s2/sqrt(E2);
