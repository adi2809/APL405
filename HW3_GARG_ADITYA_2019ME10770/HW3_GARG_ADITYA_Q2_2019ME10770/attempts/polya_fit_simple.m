function a = polya_fit_simple(data,a)
% POLYA_FIT_SIMPLE   Maximum-likelihood Polya distribution.
%
% Same as POLYA_FIT but uses the simple fixed-point iteration described in
% "Estimating a Dirichlet distribution" by T. Minka. 

show_progress = 0;

if nargin < 2
  a = polya_moment_match(data);
end
sdata = sum(data, 2);

% fixed-point iteration
[N,K] = size(data);
for iter = 1:1000
  old_a = a;
  sa = sum(a);
  if 0
%     g = col_sum(digamma(data + repmat(a, N, 1))) - N*digamma(a);   % col_sum is equivalent to sum(x,1)
    g = sum(digamma(data + repmat(a, N, 1)),1) - N*digamma(a);
    h = sum(digamma(sdata + sa)) - N*digamma(sa);
  else
%     g = col_sum(di_pochhammer(repmat(a, N, 1), data));
    g = sum(di_pochhammer(repmat(a, N, 1), data),1);
    h = sum(di_pochhammer(sa, sdata));
  end
  a = a .* g ./ h;
  if show_progress
    e(iter) = sum(polya_logProb(a, data));
  end
  if max(abs(a - old_a)) < 1e-6
    break
  end
  if show_progress & rem(iter,10) == 0
    plot(e)
    drawnow
  end
end  
if show_progress
  plot(e)
end

end

function p = polya_logProb(a, data)
% POLYA_LOGPROB   Dirichlet-multinomial (Polya) distribution.
%
% POLYA_LOGPROB(a,data) returns a vector containing the log-probability of 
% each histogram in DATA, under the Polya distribution with parameter A.
% DATA is a matrix of histograms.
% If A is a row vector, then the histograms are the rows, otherwise columns.

if any(a < 0)
  p = -Inf;
  return
end
row = (rows(a) == 1);

s = full(sum(a));
if row
  sdata = sum(data,2);
  p = zeros(rows(data),1);
  for k = 1:cols(data)
    dk = data(:,k);
    p = p + pochhammer(a(k), dk);
  end
  p = p - pochhammer(s, sdata);
else
  sdata = col_sum(data);
  for i = 1:cols(data)
    p(i) = sum(gammaln(data(:,i) + a)) - gammaln(sdata(i) + s);
  end
  p = p + gammaln(s) - sum(gammaln(a));
end

end

function y = pochhammer(x,n)
% pochhammer(x,n) returns the rising log-factorial log(gamma(x+n)/gamma(x))
% Named after the corresponding Mathematica function.
%
% pochhammer.c provides a faster implementation.

if 0 && length(x) == 1 && all(n < 100)
  nmax = full(max(max(n)));
  t(1) = 0;
  y = 0;
  for i = 1:nmax
    y = y + log(x);
    t(i+1) = y;
    x = x + 1;
  end
  y = t(n+1);
  % workaround matlab's silly rules for matrix indexing
  if cols(n) == 1 & rows(y) == 1
    y = y';
  end
  return
end
if issparse(n)
  y = sparse(rows(n),cols(n));
else
  y = zeros(size(n));
end
i = (n > 0);
if length(x) == 1
  y(i) = gammaln(x+n(i)) - gammaln(x);
else
  y(i) = gammaln(x(i)+n(i)) - gammaln(x(i));
end

end

function a = polya_moment_match(data)
% DATA is a matrix of count vectors (rows)

sdata = sum(data, 2);
p = data ./ repmat(sdata+eps,1,size(data, 2));
a = dirichlet_moment_match(p);

end

function a = dirichlet_moment_match(p)
% Each row of p is a multivariate observation on the probability simplex.

a = mean(p);
m2 = mean(p.*p);
ok = (a > 0);
s = (a(ok) - m2(ok)) ./ (m2(ok) - a(ok).^2);
% each dimension of p gives an independent estimate of s, so take the median.
s = median(s);
a = a*s;

end

function y = di_pochhammer(x,n)
% di_pochhammer(x,n) returns digamma(x+n)-digamma(x), 
% with special attention to the case n==0.
%
% di_pochhammer.c provides a faster implementation.

if issparse(n)
  y = sparse(rows(n),cols(n));
else
  y = zeros(size(n));
end
i = (n > 0);
if length(x) == 1
  y(i) = digamma(x+n(i)) - digamma(x);
else
  y(i) = digamma(x(i)+n(i)) - digamma(x(i));
end

end

function y = digamma(x)
% Digamma function: d/dx log gamma(x)
% DIGAMMA(X) returns digamma(x) = d log(gamma(x)) / dx
% If X is a matrix, returns the digamma function evaluated at each element.

% This file is from pmtk3.googlecode.com


y = psi(x); % built-in mex function
end