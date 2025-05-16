clear all
clc
% Title
% Author
% Organization
% Date

% State
syms x1 x2 real
f = 1/2*x1^2+x2^2;
g = x1 + x2 - 1;
lambda_init = 0;
x_init = [0.5, 0.5];
% Parameters
toll = 1e-3;
approx = 0;
R = [x1; sqrt(2)*x2];
% [x_sol, lambda_sol] = nlp_solver_exact_Hessian(toll, x_init, lambda_init, g, f)
% [x_sol, lambda_sol] = nlp_solver_approx_Hessian(toll, x_init, lambda_init, g, f, approx, R)
[x_sol, lambda_sol] = nlp_solver_approx_Hessian_linear_search(toll, x_init, lambda_init, g, f, approx)