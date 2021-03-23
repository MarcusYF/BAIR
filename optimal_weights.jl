# COMPUTING THE OPTIMAL WEIGHTS
using DelimitedFiles

function d(p,q)
  res=0
  if (p!=q)
     if (p<=0) p = eps() end
     if (p>=1) p = 1-eps() end
     res=(p*log(p/q) + (1-p)*log((1-p)/(1-q)))
  end
  return(res)
end

function dicoSolve(f, xMin, xMax, delta=1e-11)
  # find m such that f(m)=0 using dichotomix search
  l = xMin
  u = xMax
  sgn = f(xMin)
  while u-l>delta
    m = (u+l)/2
    if f(m)*sgn>0
      l = m
    else
      u = m
    end
  end
  m = (u+l)/2
  return m
end

function I(alpha,mu1,mu2)
    if (alpha==0)|(alpha==1)
       return 0
    else
        mid=alpha*mu1 + (1-alpha)*mu2
        return alpha*d(mu1,mid)+(1-alpha)*d(mu2,mid)
    end
end

muddle(mu1, mu2, nu1, nu2) = (nu1*mu1 + nu2*mu2)/(nu1+nu2)

function cost(mu1, mu2, nu1, nu2)
  if (nu1==0)&(nu2==0)
     return 0
  else
     alpha=nu1/(nu1+nu2)
     return((nu1 + nu2)*I(alpha,mu1,mu2))
  end
end

function xkofy(y, k, mu, delta = 1e-11)
  # return x_k(y), i.e. finds x such that g_k(x)=y
  g(x)=(1+x)*cost(mu[1], mu[k], 1/(1+x), x/(1+x))-y
  xMax=1
  while g(xMax)<0
       xMax=2*xMax
  end
  return dicoSolve(x->g(x), 0, xMax, 1e-11)
end

function aux(y,mu)
  # returns F_mu(y) - 1
  K = length(mu)
  x = [xkofy(y, k, mu) for k in 2:K]
  m = [muddle(mu[1], mu[k], 1, x[k-1]) for k in 2:K]
  return (sum([d(mu[1],m[k-1])/(d(mu[k], m[k-1])) for k in 2:K])-1)
end


function oneStepOpt(mu, delta = 1e-11)
  yMax=0.5
  if d(mu[1], mu[2])==Inf
     # find yMax such that aux(yMax,mu)>0
     while aux(yMax,mu)<0
          yMax=yMax*2
     end
  else
     yMax=d(mu[1],mu[2])
  end
  y = dicoSolve(y->aux(y, mu), 0, yMax, delta)
  x =[xkofy(y, k, mu, delta) for k in 2:length(mu)]
  pushfirst!(x, 1)
  nuOpt = x/sum(x)
  return nuOpt[1]*y, nuOpt
end


function OptimalWeights(mu, delta=1e-11)
  # returns T*(mu) and w*(mu)
  K=length(mu)
  IndMax=findall(mu.==maximum(mu))
  L=length(IndMax)
  if (L>1)
     # multiple optimal arms
     vOpt=zeros(1,K)
     vOpt[IndMax]=1/L
     return 0,vOpt
  else
     mu=vec(mu)
     index=sortperm(mu,rev=true)
     mu=mu[index] 
     unsorted=vec(collect(1:K))
     invindex=zeros(Int,K)
     invindex[index]=unsorted 
     # one-step optim
     vOpt,NuOpt=oneStepOpt(mu,delta)
     # back to good ordering
     nuOpt=NuOpt[invindex]
     NuOpt=zeros(1,K)
     NuOpt[1,:]=nuOpt
     return vOpt,NuOpt
  end
end
