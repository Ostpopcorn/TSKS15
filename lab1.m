clear all
close all

Ts = 0.1;
Trange = [-15:Ts:15];
s1 = exp(-0.1*Trange.ˆ2);
s2 = exp(-0.1*Trange.ˆ2).*cos(Trange);
E1 = sum(abs(s1.ˆ2));
E2 = sum(abs(s2.ˆ2));
s1 = s1/sqrt(E1);
s2 = s2/sqrt(E2);
