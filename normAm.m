function [c,mv] = normAm(A,m)
%NORMAM   Estimate of 1-norm of power of matrix.矩阵幂的1范数的估计
%   NORMAM(A,m) estimates norm(A^m,1).
%   If A has nonnegative elements the estimate is exact.如果A有非负元素，则估计是精确的。
%   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
%   matrix-vector products computed involving A or A^*.包含A或A^*的矩阵向量乘积 

%   Reference: A. H. Al-Mohy and N. J. Higham, Computing the actionf 计算矩阵指数的作用
%   the matrix exponential, with an application to exponential o矩阵指数，与指数o积分器的应用  
%   integrators. MIMS EPrint 2010.30, The University of Manchester, 2010.

%   Awad H. Al-Mohy and Nicholas J. Higham, March 18, 2010.

n = length(A);
if isequal(A,abs(A))
    e = ones(n,1);
    for j=1:m         % for positive matrices only 只适用于正矩阵
        e = A'*e;
    end
    c = norm(e,inf);
    mv = m;
else
    [c,v,w,it] = normest1(@afun_power);
    mv = it(2)*2*m; % Since t = 2.
end

  function Z = afun_power(flag,X)
       %AFUN_POWER  Function to evaluate matrix products needed by NORMEST1.
       %函数来评估NORMEST1所需的矩阵乘积。
       if isequal(flag,'dim')
          Z = n;
       elseif isequal(flag,'real')
          Z = isreal(A);
       else

          [p,q] = size(X);
          if p ~= n, error('Dimension mismatch'), end

          if isequal(flag,'notransp')
             for i = 1:m, X = A*X; end
          elseif isequal(flag,'transp')
             for i = 1:m, X = A'*X; end
          end

          Z = X;

       end

  end
end
