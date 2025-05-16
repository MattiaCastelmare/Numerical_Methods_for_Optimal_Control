clear all
clc

syms x1 x2 real
x = [x1; x2];
one_vector = ones(2,1);

f = x'*x + one_vector'*x;
g = x'*x - 1;

toll = 10e-8;
x_init = [-1,1];
lambda_init = 0;
approx = 0;

[x_sol, lambda_sol] = nlp_solver_approx_Hessian_linear_search(toll, x_init, lambda_init, g, f, approx)
