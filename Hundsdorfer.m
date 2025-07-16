function yout = Hundsdorfer(tspan, y0, numsteps, A0, A1, A2, b, theta, sigma)
% CRAIGS - Hundsdorfer and Verwer ADI method of order 2 for ODEs of the type 
%这种类型的ode的2阶的Hundsdorfer和Verwer ADI方法  
%          y_t = A y + b 
%
% SYNOPSIS:
%   YOUT = HUNDSDORFER(TSPAN, Y0, NUMSTEPS, AO, A1, A2, b, theta, sigma) 
%
% PARAMETERS: 
%   TSPAN    - is a vector containing the endpoints of the time integration.一个向量是否包含时间积分的端点 
%   Y0       - is a vector of initial conditions.是一个初始条件的向量。
%   NUMSTEPS - number of steps to be taken, fixed stepsize.要采取的步骤数，固定的步骤大小
%   A0       - matrix relating to the discretization in S variable.与S变量离散化相关的矩阵。
%   A1       - matrix relating to the discretization in v variable.与v变量的离散化有关的矩阵
%   A2       - matrix relating to the discretization from mixed terms.与混合项离散化有关的矩阵  
%   b        - vector of boundary conditions.边界条件向量。
%   theta    - method parameter.方法参数
%   sigma    - method parameter.
%
% RETURNS:
%   YOUT     - the numerical approximation.

dt = (tspan(2)-tspan(1))/numsteps;
Id = speye(size(A0));
A = A0+A1+A2;
mat1 = Id - theta*dt*A1;
[L1, U1, P1, Q1, R1] = lu(mat1);
mat2 = Id - theta*dt*A2;
[L2, U2, P2, Q2, R2] = lu(mat2);

Up = y0;
for i = 1:numsteps
  Y0 = Up + dt*A*Up + dt*b;
  temp1 = Y0 - theta*dt*A1*Up;
  Y1 = Q1 * (U1 \ (L1 \ (P1 * (R1 \ temp1))));
  temp2 = Y1 - theta*dt*A2*Up;
  Y2 = Q2 * (U2 \ (L2 \ (P2 * (R2 \ temp2))));
  Y0t = Y0 + sigma*dt*A*(Y2 - Up);
  temp1 = Y0t - theta*dt*A1*Y2;
  Y1t = Q1 * (U1 \ (L1 \ (P1 * (R1 \ temp1))));
  temp2 = Y1t - theta*dt*A2*Y2;
  Up = Q2 * (U2 \ (L2 \ (P2 * (R2 \ temp2))));
end 
yout = Up;
