

function yout = Douglas(tspan, y0, numsteps, A0, A1, A2, b, theta)
% DOUGLAS - ADI method of order 2 for ODEs of the type 
%
%           y_t = A y + b 
%
% SYNOPSIS:
%   YOUT = DOUGLAS(TSPAN, Y0, NUMSTEPS, AO, A1, A2, b, theta) 
%
% PARAMETERS: 
%   TSPAN    - is a vector containing the endpoints of the time integration.是一个包含时间积分端点的向量
%   Y0       - is a vector of initial conditions.是一个初始条件的向量
%   NUMSTEPS - number of steps to be taken, fixed stepsize.要采取的步骤数，固定的步骤大小
%   A0       - matrix relating to the discretization in S variable.与S变量离散化相关的矩阵 
%   A1       - matrix relating to the discretization in v variable.与v变量离散化相关的矩阵 
%   A2       - matrix relating to the discretization from mixed terms.与混合项离散化有关的矩阵
%   b        - vector of boundary conditions.边界条件向量
%   theta    - method parameter.方法参数
%
% RETURNS:
%   YOUT     - the numerical approximation.数值近似

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
  Up = Q2 * (U2 \ (L2 \ (P2 * (R2 \ temp2))));
end 
yout = Up;
