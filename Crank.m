function yout = Crank(tspan, y0, numsteps, A, b)
% CRANK - Crank-Nicolson method of order 2 for ODEs of the type 
%
%          y_t = A y + b 
%
% PARAMETERS: 
%   tspan    - is a vector containing the endpoints of the time integration.是一个包含时间积分端点的向量。  
%   y0       - is a vector of initial conditions.是一个初始条件的向量。
%   numsteps - number of steps to be taken, fixed stepsize.要采取的步骤数，固定的步骤大小。
%   A        - constant matrix.常数矩阵
%   b        - vector of boundary conditions.边界条件向量
%
% RETURNS:
%   yout     - the numerical approximation.数值近似

dt = (tspan(2)-tspan(1))/numsteps;
Id = speye(size(A));
%[m,n]=size(X）返回矩阵X的尺寸信息，并存储在m,n中，其中m中存储的是行数，n中存储的是列数
%speye(m,n)生成大小min(m,n)*min(m,n)的具有稀疏属性的方阵

% Comment this in when using recent matlab
mat = Id - dt*A/2;
[L, U, P, Q, R] = lu(mat);

Up = y0;
for i = 1:numsteps
  temp = Up + dt*A*Up/2 + dt*b;
  Up = Q * (U \ (L \ (P * (R \ temp))));
end 
yout = Up;
