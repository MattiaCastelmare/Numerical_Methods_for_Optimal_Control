function [alfa] = line_search_Armijo(M, delta_x, x)
%% Parameters
% Input:
% - beta: step size [0,1]
% - gamma: step size (<< 1/2)
% - f: cost function in symbolic form
% - delta_x: step previously calculated
% -x: actual state
% Output:
% alfa: step size for the line search
alfa = 1;
beta = 0.5;
gamma = 0.1;
vars_state = symvar(M);
Grad_f = matlabFunction(gradient(M, vars_state), "Vars",{vars_state});
M = matlabFunction(M, 'Vars', {vars_state});

while M(x+alfa*delta_x') >= M(x) + gamma*alfa*Grad_f(x)'*delta_x
    alfa = beta*alfa;
end
end