function f = rhsheston(t, y, A, b);
% RHSHESTON - Computes the full rhs of the discretised Heston PDE
%计算离散的Heston PDE的完整rhs 
%             y_t = A y + b 
%
% DESCRIPTION:
%   This is used for the matlab function ode15s.
%
% PARAMETERS: 
%   t    - is a scalar of current time.是当前时间的标量。
%   y    - is a vector of solution values.是解值的向量
%   A    - constant matrix.
%   b    - vector of boundary conditions.
%
% RETURNS:
%   f    - the rhs.

f = A*y+b;
