clear all
close all

Ts = 0.1;
Trange = [-15:Ts:15];
s1 = exp(-0.1*Trange.^2);
s2 = exp(-0.1*Trange.^2).*cos(Trange);
E1 = sum(abs(s1.^2));
E2 = sum(abs(s2.^2));
s1 = s1/sqrt(E1);
s2 = s2/sqrt(E2);
