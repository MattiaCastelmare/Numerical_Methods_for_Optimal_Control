function [x_sol, lambda_sol] = nlp_solver_approx_Hessian_linear_search(toll, x_init, lambda_init, g, f, approx, varargin)
%% Parameters
% Input:
% - toll: for the comparison
% - x_init: initial_guess for the state
% - lambda_init: initial value of the Lagrangian multiplier
% - g: equality constraint
% - f: cost function
% - approx: flag to use approximation of the Hessian (0 to use exact
% Hessian and 1 to use the approximation of it)
% - varargin: f(x) = 1/2*||R(x)||^2 residual function 
% Output:
% - x_sol: solution for the state
% - lamdba_sol: solution for lambda

if length(varargin) >= 1
    R = varargin{1};
end
%% Calculate Lagrangian and Gradient
syms lambda real

L = f + lambda'*g;
vars_state = symvar(f);
vars = [symvar(f), lambda];

% x = sym("x",size(x_init));
Grad_L = matlabFunction(gradient(L, vars_state), 'Vars',{vars});

g_size = size(g,1);
sigma = 0.3;
M = f + sigma*norm(g,1);
g_toll = matlabFunction(g, 'Vars',{vars});

block_zeros = zeros(g_size, g_size);

Hessian_L = matlabFunction(hessian(L, vars_state), "Vars",{vars});
Grad_g = matlabFunction(gradient(g, vars_state), "Vars",{vars});
if approx
    disp("Hessian approximation")
    J = jacobian(R, vars_state);
    Hessian_L = J.'*J;
    Hessian_L = matlabFunction(Hessian_L, "Vars",{vars});
else
    disp("Exact Hessian")
end
Grad_f = matlabFunction(gradient(f, vars_state), "Vars",{vars});
iteration = 1;
f_print = matlabFunction(f, "Vars",{vars});
x_history = [];
violations = [];
alphas = [];
%% Main Loop
while norm(Grad_L([x_init, lambda_init]), inf) >= toll || norm(g_toll([x_init, lambda_init]), inf) >= toll

    fprintf("Iteration number: %d\n", iteration);
    fprintf("Cost: %.3f\n", f_print(x_init));
    fprintf("Tollerance: %.3f\n", toll)
    fprintf("KKT violation for Grad_L: %.3f\n", norm(Grad_L([x_init, lambda_init])));
    fprintf("KKT violation for g: %.3f\n", norm(g_toll([x_init, lambda_init])));
    
    % Salva il punto attuale
    x_history(:, end+1) = x_init(:);  % ogni colonna è un punto
    
    % Plot dinamico
    figure(1);
    hold on;
    grid on;
    
    % Disegna il vincolo (cerchio unitario) solo alla prima iterazione
    if iteration == 1
        theta = linspace(0, 2*pi, 300);
        plot(cos(theta), sin(theta), 'r', 'DisplayName', 'Constraint: ||x|| = 1');
    end
    
    % Unisci i punti con una linea blu
    if size(x_history, 2) > 1
        plot(x_history(1, end-1:end), x_history(2, end-1:end), 'b.-', ...
             'LineWidth', 1.5, 'MarkerSize', 15, 'DisplayName', 'x trajectory');
    else
        plot(x_init(1), x_init(2), 'bo', 'MarkerFaceColor', 'b');
    end
    
    axis equal;
    xlim([-2, 2]);
    ylim([-2, 2]);
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    title('Trajectory of $x$ during iterations', 'Interpreter', 'latex');

    kkt_violation = norm([Grad_L([x_init, lambda_init]); g_toll([x_init, lambda_init])], inf);
    violations(end+1) = kkt_violation;
    

    A = [Hessian_L([x_init, lambda_init]), Grad_g([x_init, lambda_init]);
         Grad_g([x_init, lambda_init])', block_zeros];
    
    b = - [Grad_f([x_init, lambda_init]);
            g_toll([x_init, lambda_init])];

    sol = A \ b;
    
    delta_x = sol(1:size(x_init,2),:);
    
    alfa = line_search_Armijo(M, delta_x, x_init);
    alphas(end+1) = alfa;
    fprintf("Alfa after linesearch: %.3f\n", alfa);
    x_init = x_init + alfa*delta_x';
    lambda_init = (1 - alfa)*lambda_init + alfa*sol((size(x_init,2)+1):end,:);

    iteration = iteration +1;
end

x_sol = x_init;
lambda_sol = lambda_init;


figure(2);
semilogy(1:length(violations), violations, 'b-o', 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('$\|[\nabla L; g]\|_\infty$', 'Interpreter', 'latex');
title('KKT Violation (Infinity Norm) over Iterations');

figure(3)
plot(1:length(alphas), alphas, 'r-o', 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('$\alpha$', 'Interpreter', 'latex');
title('Line Search Step Size $\alpha$ over Iterations', 'Interpreter', 'latex');
end