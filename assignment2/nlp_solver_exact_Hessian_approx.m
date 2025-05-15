function [x_sol, lambda_sol] = nlp_solver_exact_Hessian(toll, x_init, lambda_init, g, f, approx)
%% Variables
% Input:
% - toll: for the comparison
% - x_init: initial_guess for the state
% - lambda_init: initial value of the Lagrangian multiplier
% - g: equality constraint
% - f: cost function
% Output:
% - x_sol: solution for the state
% - lamdba_sol: solution for lambda

%% Calculate Lagrangian and Gradient
syms lambda real

L = f + lambda'*g;
vars_state = symvar(f);
vars = [symvar(f), lambda];

% x = sym("x",size(x_init));
Grad_L = matlabFunction(gradient(L, vars_state), 'Vars',{vars});

g_size = size(g,1);
g_toll = matlabFunction(g, 'Vars',{vars});

block_zeros = zeros(g_size, g_size);

Hessian_L = matlabFunction(hessian(L, vars_state), "Vars",{vars});
Grad_g = matlabFunction(gradient(g, vars_state), "Vars",{vars});
if approx
    syms x1 x2 real
    R = [x1; sqrt(2)*x2];
    J = jacobian(R, vars_state);
    Hessian_L = J.'*J;
    Hessian_L = matlabFunction(Hessian_L, "Vars",{vars});
end
Grad_f = matlabFunction(gradient(f, vars_state), "Vars",{vars});
iteration = 1;

%% Main Loop
while norm(Grad_L([x_init, lambda_init])) >= toll || norm(g_toll([x_init, lambda_init])) >= toll
    fprintf("Iteration number: %d\n", iteration);

    A = [Hessian_L([x_init, lambda_init]), Grad_g([x_init, lambda_init]);
         Grad_g([x_init, lambda_init])', block_zeros];
    
    b = - [Grad_f([x_init, lambda_init]);
            g_toll([x_init, lambda_init])];

    sol = A \ b;

    delta_x = sol(1:size(x_init,2),:);
    x_init = x_init + delta_x';
    lambda_init = sol((size(x_init,2)+1):end,:);

    iteration = iteration +1;
   
end
x_sol = x_init;
lambda_sol = lambda_init;
end