
rm(list=ls())
source("./funs.R")
library(glmnet)

##### Poisson model #####
# n         initial sample size
# p         dimension
# s         sparsity
# AR        Toeplitz matrix AR^{|i-j|}
# strength  signal strength of nonzero entry
# M         total number of learning batch
# coeff     sample inflation factor, j-th batch contain n*coeff^{j-1} sample
# eta1      learn rate for ADIHT, ADLasso, RenewLasso, RenewIHT
# eta2      learn rate for OSIM
# card      number of alternative hyperparameters for each batch
# kappa     decay rate in IHT-type method
# Con       penalty coefficient in Massart-type IC 
# myseed    your seed
#
OnlinePoisson = function(n = 100, p = 1000, s=10, AR = 0.5,
                         strength=0.5, M = 10, coeff=1,
                         eta1=1/100, eta2 =1/100, card=9,
                         kappa = 0.75, Con=0.8, myseed = 123){ 
  
  # Preliminary
  g_fun    = function(x) return( exp(x) )
  g_prime  = function(x) return( exp(x) )
  g_second = function(x) return( exp(x) )
  
  set.seed(myseed)
  
  ### first batch 
  
  beta = rep(0, p); beta[1:s] = strength* (-1)^(1:s)
  X = rmvn(n,rep(0,p),toeplitz(AR^((1:p)-1)))
  lamy = exp(X %*% beta)  
  Y = rpois(n, lamy)
  Nnew = n;
  
  ##### 1.ADIHT #####
  # choose first-batch estimation via Massart-type IC
  lamlist = 10^seq(-1.5,-0.5,l=card); 
  IClist=rep(0,card ); TempBeta = matrix(0, p, card) ; real = rep(0,card)
  melist = rep(0, card)
  for( lam in 1:card ){
    tempb = ADIHT.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                         lambda_inf = lamlist[lam], etaloc = F, 
                         kappa = kappa, eta = eta1, roundcoef=50 )
    melist[lam] = tempb$eta
    tempb = tempb$beta_new
    TempBeta[,lam] = tempb;  tempre = X %*% tempb
    IClist[lam] = sum( exp(tempre) ) - t(Y) %*% tempre + 
      Con*max(sum(tempb!=0),2)*log(p) 
    real[lam] =  sum( (beta -tempb)^2 )
    
  }
  mylambda = lamlist[ which.min(IClist)[1] ]
  TempBeta[1:10,]; lamlist;  IClist; real; melist
  apply(TempBeta, 2, function(x) sum(x!=0))
  
  # use the proper lambda for learning
  learn0 = ADIHT.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                        lambda_inf = mylambda, kappa = kappa, eta = eta1,
                        etaloc = F, roundcoef=50 )
  learning = learn0  
  
  
  ##### 2,3. Initial lasso for ADLasso and RenewLasso #####
  # learn via 5-fold cv
  myfitlasso = cv.glmnet(X, Y, nfolds=5, intercept = F,
                         family="poisson"  ) 
  
  adla = rela = coef(myfitlasso)[2:(p+1)]
  laminf = myfitlasso[["lambda.1se"]]
  sum( (beta -adla)^2 )
  adla[1:10]
  
  gradb = t(X) %*% ( g_prime(X%*%adla) - Y )
  Hessb = t(X) %*% diag( c(g_second(X%*%adla)) ) %*% X 
  Internew = gradb - (Hessb %*% adla) 
  
  # return ADLasso list
  learnLasso = list( beta_old = rep(0,p),  beta_new = adla,   
                     Inter_new= Internew,  Hess_new = Hessb, 
                     N_old    = 0,         N_new    = Nnew, 
                     laminf   = laminf  )
  
  # return RenewLasso list
  Renewlasso = list(beta_old = rep(0,p),  beta_new = rela,      
                    Hess_new = Hessb, 
                    N_old    = 0,         N_new    = Nnew,
                    laminf   = laminf  )
  
  # Design a sequence set based on laminf
  # decreasing sequence!!!
  lams = 10^seq( log10(laminf)+0.5, log10(laminf)-0.5, l=card)
  candilasso = glmnet(X, Y, intercept = F, family="poisson",
                      lambda = lams, thresh=1e-3) 
  
  if( dim(candilasso[["beta"]])[2] == card){
    RLcandi = list(lamcandi = lams,
                   estcandi = candilasso[["beta"]])
  }else{
    RLcandi = list(lamcandi = lams,
                   estcandi =matrix(0, p, card) )
  }
  
  apply(RLcandi[["estcandi"]], 2, function(x) sum( (beta -x)^2 )  )
  
  # candilasso[["beta"]][1:10,]
  # adla[1:10]
  
  
  ##### 4.RenewIHT #####
  # choose first-batch estimation via Massart-type IC
  lamlist = 10^seq(-2,-1,l=card); 
  IClist=rep(0,card ); TempBeta = matrix(0, p, card) #; real = rep(0,card)
  
  for( lam in 1:card ){
    tempb = Renew.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                         warm = FALSE, beta_ini = rep(0,p), Hess = NULL, N = 0,
                         lambda_inf = lamlist[lam], kappa = kappa,
                         eta=eta1, method = "IHT", roundcoef =30 )$beta_new
    TempBeta[,lam] = tempb;  tempre = X %*% tempb
    IClist[lam] = sum( exp(tempre) ) - t(Y) %*% tempre +
      Con * max(sum(tempb!=0), 2) * log(p)
    # real[lam] =  sum( (beta -tempb)^2 )
  }
  
  mylambda = lamlist[ which.min(IClist)[1] ]
  # TempBeta[1:15,];  IClist;  which(TempBeta[,7]!=0) 
  
  # use the proper lambda for learning
  RenewIHT = Renew.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                          warm = FALSE, beta_ini = rep(0,p), Hess = NULL, N = 0,
                          lambda_inf = mylambda, eta=eta1, kappa = kappa,
                          method = "IHT", roundcoef =30)
  # which(RenewIHT$beta_new!=0)
  
  
  ##### 5.OSIM #####
  # choose first-batch estimation via Massart-type IC
  lamlist = 10^seq(-1.5,-0,l=card); 
  IClist=rep(0,card ); TempBeta = matrix(0, p, card) ; real = rep(0,card); Ti = list()
  
  part1 = 1:(n/2);  part2 = (n/2+1):n
  X1 = X[part1, ];     Y1 = Y[part1]
  X2 = X[part2, ];     Y2 = Y[part2]
  
  for( lam in 1:card ){
    tempb = OSIM(X=X, Y=Y, g_prime=g_prime, g_second=g_second, warm = FALSE,
                 beta_ini1 = rep(0,p), Hess1 = NULL,
                 beta_ini2 = rep(0,p), Hess2 = NULL,
                 N = 0, lambda_inf = lamlist[lam], roundcoef =50, eta= eta2) #double gradient, half learn
    Ti[[lam]] = tempb$iter
    TempBeta[,lam] =  tempb$beta_ave 
    tempre1 = X1 %*% tempb[["beta_new1"]];  tempre2 = X2 %*% tempb[["beta_new2"]]
    loss1 = 2*( sum( exp(tempre1) ) - t(Y1) %*% tempre1 )
    loss2 = 2*( sum( exp(tempre2) ) - t(Y2) %*% tempre2 )
    IClist[lam] = loss1 + loss2 + 
      Con/5*( sum(tempb[["beta_new1"]]!=0) + sum(tempb[["beta_new2"]]!=0) )*log(p)
    
    real[lam] =  sum( (beta - tempb$beta_ave )^2 )
  }
  
  mylambda = lamlist[ which.min(IClist)[1] ]
  # TempBeta[1:10,];  IClist; real; 
  
  # use the proper lambda for learning
  RenewOSIM = OSIM(X=X, Y=Y, g_prime=g_prime, g_second=g_second, warm = FALSE, 
                   beta_ini1 = rep(0,p), Hess1 = NULL,
                   beta_ini2 = rep(0,p), Hess2 = NULL,
                   N = 0, lambda_inf = mylambda, roundcoef =50, eta=eta2) 
  
  
  
  ##### 6.RADAR #####
  # Rk, alphak used values provided in Han2026JASA
  lamlist = 10^seq(-4,-1,l=card); 
  LogitRADAR = RADAR( X=X, Y=Y, g_fun=g_fun, g_prime=g_prime, 
                      betapast = rep(1e-6,p), 
                      lastlambdalist = lamlist,
                      betapastlist = matrix(0, p, card),
                      mylambdalist = lamlist,
                      Rk=1.2*sum( abs(beta) ), alphak=10)
  
  
  ##### 7.Oracle #####
  Xf = X[,1:s]; Yf = Y
  betaora = rep(0,p)
  betaora[1:s] = glm(Yf ~ Xf+0, family = poisson(link = "log") )[["coefficients"]]
  sum( (beta - betaora)^2 )
  
  ##### subsequent batch #####
  # estimator collection
  BetaADIHT = BetaADlasso = BetaRenewlasso = BetaRenewIHT = BetaOsim = BetaRADAR = BetaOracle = matrix(0, p, M)
  BetaADIHT[,1]      = learning$beta_new
  BetaADlasso[,1]    = learnLasso[["beta_new"]]
  BetaRenewlasso[,1] = Renewlasso[["beta_new"]]
  BetaRenewIHT[,1]   = RenewIHT[["beta_new"]]
  BetaOsim[,1]       = RenewOSIM[["beta_ave"]]
  BetaRADAR[,1]      = LogitRADAR[["betahat"]]
  BetaOracle[,1]     = betaora
  
  for( k in 2:M){
    n= round(n*coeff)
    X = rmvn(n,rep(0,p),toeplitz(AR^((1:p)-1)))
    lamy = exp(X %*% beta)  
    Y = rpois(n, lamy)
    Npast = Nnew;         # the cumulative sample size BEFORE this batch learning
    Nnew = Npast + n      # the cumulative sample size AFTER  this batch learning
    
    ##### 1.ADIHT ##### 
    Inter = learning$Inter_new;  Hess = learning$Hess_new
    mylambda = learning[["laminf"]]*sqrt(Npast/Nnew)
    
    # adaptive tuning with Massart-type IC
    lamlist = 10^seq( log10(mylambda)-0.5, log10(mylambda)+0.5, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) ; real = rep(0,card)
    
    for( lam in 1:card ){
      tempb = ADIHT.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                           warm = FALSE, Inter = Inter, Hess = Hess,
                           lambda_inf = lamlist[lam], N = Npast, 
                           kappa = kappa, eta = eta1, roundcoef=50 )$beta_new
      TempBeta[,lam] = tempb;  tempre = X %*% tempb
      IClist[lam] = sum( exp(tempre) ) - t(Y) %*% tempre +
        t(tempb) %*% Inter + 0.5*t(tempb) %*%Hess %*%tempb + 
        Con *max(sum(tempb!=0),2) * log(p)  
      real[lam] =  sum( (beta -tempb)^2 )
    } 
    mylambda = lamlist[ which.min(IClist)[1] ] 
    TempBeta[1:10,];  IClist; real
    apply(TempBeta, 2, function(x) sum(x!=0))
    # which(TempBeta[,which.min(IClist)[1]]!=0); 
    # which(TempBeta[,2]!=0); 
    # TempBeta[which(TempBeta[,which.min(IClist)[1]]!=0), which.min(IClist)[1] ]
    
    # b-th batch learning
    learning = ADIHT.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second, 
                            warm = FALSE, Inter = Inter, Hess = Hess, roundcoef=50,
                            lambda_inf = mylambda, N = Npast, kappa = kappa, eta=eta1)
    BetaADIHT[,k] = learning$beta_new
    
    
    ##### 2.ADLasso ##### 
    Inter = learnLasso$Inter_new; Hess = learnLasso$Hess_new
    mylambda = learnLasso[["laminf"]] * sqrt(Npast / Nnew)
    
    # adaptive tuning with Massart-type IC
    lamlist = 10^seq( log10(mylambda)-0.5, log10(mylambda)+0.5, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) ; real = rep(0,card)
    
    for( lam in 1:card ){
      tempb = ADIHT.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second, warm = FALSE,
                           Inter = Inter, Hess = Hess, lambda_inf = lamlist[lam],
                           N = Npast, eta=eta1, roundcoef=50,
                           method="lasso" )$beta_new
      TempBeta[,lam] = tempb;  tempre = X %*% tempb
      IClist[lam] = sum( exp(tempre)  ) - t(Y) %*% tempre +
        t(tempb) %*% Inter + 0.5*t(tempb) %*%Hess %*%tempb + 
        Con*sum(tempb!=0)*log(p)/5  
    }
    mylambda = lamlist[ which.min(IClist)[1] ]
    # TempBeta[1:15,];  IClist; 
    
    # b-th batch learning
    learnLasso = ADIHT.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                              warm = FALSE,  Inter = Inter, Hess = Hess,
                              lambda_inf = mylambda, N = Npast, eta=eta1,
                              method="lasso", roundcoef=50 )
    BetaADlasso[,k] = learnLasso$beta_new
    
    
    
    ##### 3.RenewLasso ##### 
    Hess = Renewlasso[["Hess_new"]]; lastbeta = Renewlasso[["beta_new"]]
    
    ## get optimal lambda_b follows Luo2023EJS
    tempMSPE = rep(0, card)
    for( idx in 1:card){
      tempMSPE[idx] = sum( (Y - g_prime( X %*% RLcandi[["estcandi"]][,idx] ))^2 )
    }
    optidx = which.min(tempMSPE)[1]
    optlam = RLcandi[["lamcandi"]][optidx]
    
    Renewlasso = Renew.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                              warm = FALSE, beta_ini = lastbeta, Hess = Hess,
                              N = Npast, lambda_inf = optlam, 
                              eta=eta1, method = "lasso", roundcoef=50 )
    # Renewlasso[["beta_new"]][1:15]
    
    BetaRenewlasso[,k]  = Renewlasso[["beta_new"]]
    
    ## update candidate lambda list and beta list
    RLcandi[["lamcandi"]] = 10^seq( log10(optlam)+0.5, log10(optlam)-0.5, l=card) 
    
    ttl = matrix(0, p, card)
    for ( idx in 1: card){
      ttl[,idx] = Renew.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                               warm = FALSE, beta_ini = lastbeta, Hess = Hess,
                               N = Npast, lambda_inf = RLcandi[["lamcandi"]][idx],
                               eta=eta1, method="lasso", roundcoef=50 )[["beta_new"]]
    }
    RLcandi[["estcandi"]] = ttl
    
    
    
    ##### 4.RenewIHT #####
    Hess = RenewIHT[["Hess_new"]]; lastbeta = BetaRenewIHT[,k-1]
    mylambda = RenewIHT[["laminf"]]*sqrt(Npast/Nnew)
    
    # adaptive tuning with Massart-type IC
    lamlist = 10^seq( log10(mylambda)-0.5, log10(mylambda)+0.5, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) ; real = rep(0,card)
    
    for( lam in 1:card ){
      tempb = Renew.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                           warm = FALSE, beta_ini = lastbeta, Hess = Hess,
                           N = Npast, lambda_inf = lamlist[lam], kappa=kappa,
                           eta=eta1, method = "IHT", roundcoef= 30 )$beta_new
      TempBeta[,lam] = tempb;  tempre = X %*% tempb
      IClist[lam] = sum( exp(tempre)  ) - t(Y) %*% tempre + 
        0.5*t(tempb) %*% Hess %*%tempb + Con*max(sum(tempb!=0),2)*log(p)  
    } 
    mylambda = lamlist[ which.min(IClist)[1] ]
    # TempBeta[1:15,];  IClist; which(TempBeta[,4]!=0) 
    
    RenewIHT = Renew.online(X=X, Y=Y, g_prime=g_prime, g_second=g_second,
                            warm = FALSE, beta_ini = lastbeta, kappa=kappa,
                            Hess = Hess, N = Npast, lambda_inf = mylambda,
                            eta = eta1, method = "IHT", roundcoef= 30 )
    BetaRenewIHT[,k]  = RenewIHT[["beta_new"]]
    
    
    ##### 5.OSIM #####
    Hess1 = RenewOSIM[["Hess_new1"]];     Hess2 = RenewOSIM[["Hess_new2"]]; 
    lastbeta1 = RenewOSIM[["beta_new1"]]; lastbeta2 = RenewOSIM[["beta_new2"]]; 
    mylambda = RenewOSIM[["laminf"]]*sqrt(Npast/Nnew)
    
    # adaptive tuning with Massart-type IC
    lamlist = 10^seq( log10(mylambda)-0.5, log10(mylambda)+0.5, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) #; real = rep(0,card)
    
    part1 = 1:(n/2);  part2 = (n/2+1):n
    X1 = X[part1, ];     Y1 = Y[part1]
    X2 = X[part2, ];     Y2 = Y[part2]
    
    for( lam in 1:card ){
      tempb = OSIM(X=X, Y=Y, g_prime=g_prime, g_second=g_second, warm = FALSE,
                   beta_ini1 = lastbeta1,  Hess1 = Hess1,
                   beta_ini2 = lastbeta2,  Hess2 = Hess2, 
                   N = Npast, lambda_inf = lamlist[lam], 
                   eta=eta2, method = "lasso", roundcoef = 50 ) #double gradient, half learn
      TempBeta[,lam] =  tempb$beta_ave
      b1 = tempb[["beta_new1"]];   tempre1 = X1 %*% b1
      b2 = tempb[["beta_new2"]];   tempre2 = X2 %*% b2
      loss1 = 2*( sum( exp(tempre1) ) - t(Y1) %*% tempre1 ) +
        0.5* t(b1-lastbeta2) %*% Hess1 %*% (b1-lastbeta2)
      loss2 = 2*( sum( exp(tempre2) ) - t(Y2) %*% tempre2 ) +
        0.5* t(b2-lastbeta1) %*% Hess2 %*% (b2-lastbeta1)
      IClist[lam] = loss1 + loss2 + Con/5*max( ( sum(b1!=0) + sum(b2!=0) ), 2)*log(p)
    }
    mylambda = lamlist[ which.min(IClist)[1] ]
    # TempBeta[1:15,];  IClist;  which(TempBeta[,1]!=0)  
    RenewOSIM = OSIM(X=X, Y=Y, g_prime=g_prime, g_second=g_second, warm = FALSE, 
                     beta_ini1 = lastbeta1,  Hess1 = Hess1,
                     beta_ini2 = lastbeta2,  Hess2 = Hess2, roundcoef = 50,
                     N = Npast, lambda_inf = mylambda, eta=eta2, method = "lasso")
    BetaOsim[,k]  = RenewOSIM[["beta_ave"]]
    
    
    
    ##### 6.RADAR #####
    #Rk, alphak update guided by Han2026JASA
    # Rk               = LogitRADAR[["usedRk"]]/sqrt(coeff)   
    # alphak           = LogitRADAR[["usedalphak"]]*sqrt(coeff)
    Rk               = LogitRADAR[["usedRk"]]/sqrt(2)
    alphak           = LogitRADAR[["usedalphak"]]*2
    RadarLamPastList = LogitRADAR[["usedlambdalist"]]
    RadarLamNowList  = LogitRADAR[["usedlambdalist"]]*sqrt(Npast/Nnew)
    betapast         = LogitRADAR[["betahat"]]
    betapastlist     = LogitRADAR[["usedbetaseq"]]
    
    LogitRADAR = RADAR(X=X, Y=Y, g_fun=g_fun, g_prime=g_prime, 
                       betapast = betapast, lastlambdalist = RadarLamPastList,
                       betapastlist = betapastlist,
                       mylambdalist = RadarLamNowList,
                       Rk=Rk, alphak=alphak )
    
    BetaRADAR[,k] = LogitRADAR[["betahat"]]
    
    
    
    ##### 7. Oracle #####
    Xf = rbind(Xf, X[,1:s]); Yf = c(Yf, Y)
    BetaOracle[1:s,k] = glm(Yf~Xf+0, family=poisson(link = "log") )[["coefficients"]]
    
    cat(k, "/", M, "\r")
  }
  
  return( list(BetaADIHT=BetaADIHT, 
               BetaADlasso=BetaADlasso,
               BetaRenewlasso= BetaRenewlasso,
               BetaRenewIHT=BetaRenewIHT,
               BetaOsim=BetaOsim,
               BetaRADAR=BetaRADAR,
               BetaOracle=BetaOracle ) )
}




##### Fixed MC #####
PoissonFix = list()
MC = 30
for( mc in 1:MC ){
  t1 = Sys.time()
  tst = OnlinePoisson(n = 100, p = 1000, s=10, AR = 0.5, strength=0.5,
                      M = 30, coeff=1, card=9, kappa = 0.8, Con= 1,
                      eta1=0.15 , eta2=0.08, myseed = mc)
  t2 = Sys.time()
  PoissonFix[[mc]] = tst
  
  cat("time:", t2-t1, " ", mc, "/", MC, "\n")
}

save(PoissonFix, file = "Poissonfix0422.RData")
  


##### Increasing MC #####
PoissonIncrease = list()
MC = 30;   M = 12     # 12 batches learning

for( mc in 1:MC ){
  t1 = Sys.time()
  tst = OnlinePoisson(n = 100, p = 1000, s=10, AR = 0.5, strength=0.5,
                      M = M, coeff=sqrt(2), card=9, kappa = 0.8, Con=1,
                      eta1=0.15, eta2=0.08, myseed = mc+MC)
  t2 = Sys.time() 
  PoissonIncrease[[mc]] = tst
  
  # save(PoissonIncrease, file = "Poissonincrease0423.RData")
  
  cat("time:", t2-t1, " ", mc, "/", MC, "\n")
}

save(PoissonIncrease, file = "Poissonincrease0423.RData")




