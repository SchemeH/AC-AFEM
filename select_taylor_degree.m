function  [M,mv,alpha,unA] = ...
           select_taylor_degree(A,m_max,p_max,prec,shift,bal,force_estm)
%SELECT_TAYLOR_DEGREE   Select degree of Taylor approximation. 选择泰勒近似度
%   [M,MV,alpha,unA] = SELECT_TAYLOR_DEGREE(A,m_max,p_max) forms a matrix M
%   for use in determining the truncated Taylor series degree in EXPMV
%   形成矩阵M，用于确定EXPMV中的截断泰勒级数 
%   and EXPMV_TSPAN, based on parameters m_max and p_max.参数
%   MV is the number of matrix-vector products with A or A^* computed.
%   MV是与A或A^*计算的矩阵向量乘积的个数
%   Reference: A. H. Al-Mohy and N. J. Higham, Computing the action of
%   the matrix exponential, with an application to exponential
%   integrators. MIMS EPrint 2010.30, The University of Manchester, 2010.
%   计算矩阵指数的作用，并应用于指数积分器 
%   Awad H. Al-Mohy and Nicholas J. Higham, March 19, 2010.

if nargin < 7, force_estm = false; end
if nargin < 3 || isempty(p_max), p_max = 8; end
if nargin < 2 || isempty(m_max), m_max = 55; end

if p_max < 2 || m_max > 60 || m_max + 1 < p_max*(p_max - 1)
    error('>>> Invalid p_max or m_max.')
end
n = length(A);
if nargin < 6 || isempty(bal), bal = false; end
if bal
    [D B] = balance(A);
    if norm(B,1) < norm(A,1), A = B; end
end
if nargin < 4 || isempty(prec), prec = class(A); end
switch prec
    case 'double'
        load theta_taylor
    case 'single'
        load theta_taylor_single
end
if shift
    mu = trace(A)/n;
    A = A-mu*speye(n);
end
mv = 0;
if ~force_estm, normA = norm(A,1); end

if ~force_estm && normA <= 4*theta(m_max)*p_max*(p_max + 3)/m_max
    % Base choice of m on normA, not the alpha_p.
    % m在normA上的基底选择，而不是p
    unA = 1;
    c = normA;
    alpha = c*ones(p_max-1,1);
else
    unA = 0;
    eta = zeros(p_max,1); alpha = zeros(p_max-1,1);
    for p = 1:p_max
        [c,k] = normAm(A,p+1);
        c = c^(1/(p+1));
        mv = mv + k;
        eta(p) = c;
    end
    for p = 1:p_max-1
        alpha(p) = max(eta(p),eta(p+1));
    end
end
M = zeros(m_max,p_max-1);
for p = 2:p_max
    for m = p*(p-1)-1 : m_max
        M(m,p-1) = alpha(p-1)/theta(m);
    end
end
