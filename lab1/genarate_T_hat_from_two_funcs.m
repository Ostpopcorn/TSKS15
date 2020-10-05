function hat = genarate_T_hat_from_two_funcs(s,x)
    % conv får ut 601 index. Vilket blir -30 till 30 sekunder
    % varje sekund motsvarar 10 index. NOllan gör det sista samplet, därför
    % ska nollan ingå i en delmängd över origo
    T_hat_metric = conv(s,x);
    % plot(T_hat_metric)
    % För att få de mittesta -5 till 5 sekunderna plockar jag 101 index
    % från mitten. Jag ska ha ~1/6 av alla index. 
    % 25 sekunder in är index 250 (men på matlabianska är det "index" 251)
    % och sen 101 index från det blir 351. på
    % andra sidan är det då 250 index kvar, ok.
    start_index = 250 +1;
    end_index = 350+1;
    T_hat_metric = T_hat_metric(start_index:end_index);
    [max_value, max_t_index] = max(T_hat_metric);
    
    hat = -5+ 0.1*(max_t_index-1);
end