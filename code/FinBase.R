# rm(list = ls())

source("funs.R")
library(glmnet)

# Xtrain    train X
# Ytrain    train label
# eta1      learn rate for ADIHT, ADLasso, RenewLasso, RenewIHT 
# eta2      learn rate for OnlineSIM (usually half of eta1 because it split the sample in two sets)
# card      number of alternative hyperparameters for each batch
# kappa     decay rate in IHT-type method
# Con       penalty coefficient in Massart-type IC 
# myseed    your seed
#
#
# do online learning 
# also do full sample learning
# return beta in each batch
#
ADIHT_real = function( Xtrain, Ytrain,
                       eta1=1.2, card=9, kappa = 0.75, Con=0.8){ 

  Xlist = Xtrain
  Ylist = Ytrain
  M = length(Xlist)
  p = dim(Xlist[[1]])[2]
  
  # Preliminary
  g_fun    = function(x) return( log( 1+ exp(x) ) ) 
  g_prime  = function(x) return( exp(x)/(1+exp(x)) ) 
  g_second = function(x) return( exp(x)/(1+exp(x))/(1+exp(x)) )  
  
  ### first batch 
  Nnew = dim(Xlist[[1]])[1];
  Xb = Xlist[[1]]; Yb = Ylist[[1]]
  n = dim(Xlist[[1]])[1]
  
  ##### 1.ADIHT #####
  # choose first-batch estimation via Massart-type IC
  lamlist = 10^seq(-1.5,0,l=card); 
  IClist=rep(0,card ); TempBeta = matrix(0, p, card) 
  
  for( lam in 1:card ){
    tempb = ADIHT.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second,
                         lambda_inf = lamlist[lam], etaloc = F,
                         kappa = kappa, eta = eta1)$beta_new
    TempBeta[,lam] = tempb;  tempre = Xb %*% tempb
    IClist[lam] = sum( log(1+ exp(tempre) ) ) - t(Yb) %*% tempre + 
      Con*max(sum(tempb!=0),2)*log(p) 
  }
  mylambda = lamlist[ which.min(IClist)[1] ]
  apply(TempBeta,2, function(x) sum(x!=0));  IClist; 
  # indiht = which(TempBeta[,which.min(IClist)[1]]!=0)
  # TempBeta[indiht, which.min(IClist)[1] ]
  # myword[indiht]
  
  # 
  # TempBeta[ which(TempBeta[,which.min(IClist)[1]]!=0),  which.min(IClist)[1]] 
  
  # use the proper lambda for learning
  learn0 = ADIHT.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, etaloc = F,
                        lambda_inf = mylambda, kappa = kappa, eta = eta1)
  learning = learn0  
  
  
  ##### subsequent batch #####
  # estimator collection
  BetaADIHT = matrix(0, p, M)
  BetaADIHT[,1]      = learning$beta_new
  
  
  for( k in 2:M){
    Xb = Xlist[[k]]; Yb = Ylist[[k]]
    n = dim(Xlist[[k]])[1]
    
    Npast = Nnew;         # the cumulative sample size BEFORE this batch learning
    Nnew = Npast + n      # the cumulative sample size AFTER  this batch learning
    
    time1 = Sys.time()
    
    ##### 1.ADIHT ##### 
    Inter = learning$Inter_new;  Hess = learning$Hess_new; 
    eta1 = learning[["eta"]]
    mylambda = learning[["laminf"]]*sqrt(Npast/Nnew)
    
    # adaptive tuning with Massart-type IC
    lamlist = 10^seq( log10(mylambda)-0.5, log10(mylambda)+0.5, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) 
    
    for( lam in 1:card ){
      tempb = ADIHT.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second,
                           warm = FALSE, Inter = Inter, Hess = Hess,
                           lambda_inf = lamlist[lam], N = Npast,  etaloc = F,
                           kappa = kappa, eta = eta1)$beta_new
      TempBeta[,lam] = tempb;  tempre = Xb %*% tempb
      IClist[lam] = sum( log(1+ exp(tempre) ) ) - t(Yb) %*% tempre +
        t(tempb) %*% Inter + 0.5*t(tempb) %*%Hess %*%tempb + 
        Con *max(sum(tempb!=0),2) * log(p)  
    }
    mylambda = lamlist[ which.min(IClist)[1] ]
    apply(TempBeta,2, function(x) sum(x!=0));  IClist; 
    # indiht = which(TempBeta[,which.min(IClist)[1]]!=0)
    
    
    # b-th batch learning
    learning = ADIHT.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, 
                            warm = FALSE, Inter = Inter, Hess = Hess,  etaloc = F,
                            lambda_inf = mylambda, N = Npast, kappa = kappa, eta=eta1)
    BetaADIHT[,k] = learning$beta_new
    
    
    time2 = Sys.time()
    cat(k, "/", M, ", time: ", time2-time1 , "\r")
  }
  
  return( list(BetaADIHT=BetaADIHT ) )
}


Adlasso_real = function( Xtrain, Ytrain, 
                         eta1=1.3, card=7, Con=0.8){ 
  
  M = length(Xtrain)
  p = dim(Xtrain[[1]])[2]
  Xlist = Xtrain
  Ylist = Ytrain 
  
  # Preliminary
  g_fun    = function(x) return( log( 1+ exp(x) ) ) 
  g_prime  = function(x) return( exp(x)/(1+exp(x)) ) 
  g_second = function(x) return( exp(x)/(1+exp(x))/(1+exp(x)) )  
  
  ### first batch 
  Nnew = dim(Xlist[[1]])[1];
  Xb = Xlist[[1]]; Yb = Ylist[[1]]
  n = dim(Xlist[[1]])[1]
  
  
  ##### 2. Initial lasso for ADLasso #####
  # learn via 5-fold cv
  myfitlasso = cv.glmnet(Xb, Yb, nfolds=5, intercept = F,
                         lambda = 10^seq(0,-3, l=20),
                         family="binomial"  ) 
  laminf = myfitlasso[["lambda.min"]]
  inimodel = glmnet(Xb, Yb, intercept = F,
                    lambda = laminf, family="binomial"  ) 
  adla = coef(inimodel)[-1]
  
  gradb = t(Xb) %*% ( g_prime(Xb%*%adla) - Yb )
  Hessb = t(Xb) %*% diag( c(g_second(Xb%*%adla)) ) %*% Xb
  Internew = gradb - (Hessb %*% adla) 
  
  # return ADLasso list
  learnLasso = list( beta_old = rep(0,p),  beta_new = adla,   
                     Inter_new= Internew,  Hess_new = Hessb, 
                     N_old    = 0,         N_new    = Nnew, 
                     laminf   = laminf , eta = eta1 )
  
  
  ##### subsequent batch #####
  # estimator collection
  BetaADlasso = matrix(0, p, M)
  BetaADlasso[,1]    = learnLasso[["beta_new"]]
  
  for( k in 2:M){
    Xb = Xlist[[k]]; Yb = Ylist[[k]]
    n = dim(Xlist[[k]])[1]
    
    Npast = Nnew;         # the cumulative sample size BEFORE this batch learning
    Nnew = Npast + n      # the cumulative sample size AFTER  this batch learning
    
    time1 = Sys.time()
    
    ##### 2.ADLasso ##### 
    Inter = learnLasso$Inter_new; Hess = learnLasso$Hess_new; 
    eta1 = learnLasso[["eta"]] 
    mylambda = learnLasso[["laminf"]] * sqrt(Npast / Nnew)
    
    # adaptive tuning with Massart-type IC
    lamlist = 10^seq( log10(mylambda)-0.5, log10(mylambda)+0.5, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) 
    
    for( lam in 1:card ){
      tempb = ADIHT.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, warm = FALSE,
                           Inter = Inter, Hess = Hess, lambda_inf = lamlist[lam], etaloc=F, 
                           N = Npast, eta=eta1, method="lasso", roundcoef=20)$beta_new
      TempBeta[,lam] = tempb;  tempre = Xb %*% tempb
      IClist[lam] = sum( log(1 + exp(tempre) ) ) - t(Yb) %*% tempre +
        t(tempb) %*% Inter + 0.5*t(tempb) %*%Hess %*%tempb + 
        Con*sum(tempb!=0)*log(p) 
    }
    mylambda = lamlist[ which.min(IClist)[1] ]
    apply(TempBeta,2, function(x) sum(x!=0));  IClist; 
    indiht = which(TempBeta[,which.min(IClist)[1]]!=0)
    
    
    # b-th batch learning
    learnLasso = ADIHT.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second,
                              warm = FALSE,  Inter = Inter, Hess = Hess,
                              lambda_inf = mylambda, N = Npast, eta=eta1,  etaloc=F, 
                              method="lasso", roundcoef=20)
    BetaADlasso[,k] = learnLasso$beta_new
    
    time2 = Sys.time()
    
    cat(k, "/", M, ", time: ", time2-time1 , "\r") 
  }
  
  return( list(BetaADlasso=BetaADlasso) )
}


Renewlasso_real = function( Xtrain, Ytrain, 
                            eta1=1.3, card=7, Con=0.8){ 
  
  M = length(Xtrain)
  p = dim(Xtrain[[1]])[2]
  Xlist = Xtrain
  Ylist = Ytrain 
  
  # Preliminary
  g_fun    = function(x) return( log( 1+ exp(x) ) ) 
  g_prime  = function(x) return( exp(x)/(1+exp(x)) ) 
  g_second = function(x) return( exp(x)/(1+exp(x))/(1+exp(x)) )  
  
  ### first batch 
  Nnew = dim(Xlist[[1]])[1];
  Xb = Xlist[[1]]; Yb = Ylist[[1]]
  n = dim(Xlist[[1]])[1]
  
  
  ##### 3. Initial lasso for RenewLasso #####
  # learn via 5-fold cv
  myfitlasso = cv.glmnet(Xb, Yb, nfolds=5, intercept = F,
                         lambda = 10^seq(0,-3, l=20),
                         family="binomial"  ) 
  laminf = myfitlasso[["lambda.min"]]
  inimodel = glmnet(Xb, Yb, intercept = F,
                    lambda = laminf, family="binomial"  ) 
  rela = coef(inimodel)[-1]
  
  
  Hessb = t(Xb) %*% diag( c(g_second(Xb%*%rela)) ) %*% Xb
  
  # return RenewLasso list
  Renewlasso = list(beta_old = rep(0,p),  beta_new = rela,      
                    Hess_new = Hessb, 
                    N_old    = 0,         N_new    = Nnew,
                    laminf   = laminf  ,  eta      = eta1 )
  
  # Design a sequence set based on laminf
  # decreasing sequence!!!
  lams = 10^seq( log10(laminf)+0.5, log10(laminf)-0.5, l=card)
  candilasso = glmnet(Xb, Yb, intercept = F, family="binomial",
                      lambda = lams ) 
  
  RLcandi = list(lamcandi = lams, estcandi = candilasso[["beta"]])
  
  
  ##### subsequent batch #####
  # estimator collection
  BetaRenewlasso = matrix(0, p, M)
  BetaRenewlasso[,1] = Renewlasso[["beta_new"]]
  
  for( k in 2:M){
    Xb = Xlist[[k]]; Yb = Ylist[[k]]
    n = dim(Xlist[[k]])[1]
    
    Npast = Nnew;         # the cumulative sample size BEFORE this batch learning
    Nnew = Npast + n      # the cumulative sample size AFTER  this batch learning
    
    time1 = Sys.time()
    
    ##### 3.RenewLasso ##### 
    Hess = Renewlasso[["Hess_new"]]; lastbeta = Renewlasso[["beta_new"]]; 
    eta1 = Renewlasso[["eta"]]
    
    ## get optimal lambda_b follows Luo2023EJS
    tempMSPE = rep(0, card)
    for( idx in 1:card){
      tempMSPE[idx] = sum( (Yb - g_prime( Xb %*% RLcandi[["estcandi"]][,idx] ))^2 )
    }
    optidx = which.min(tempMSPE)[1]
    optlam = RLcandi[["lamcandi"]][optidx]
    
    Renewlasso = Renew.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second,
                              warm = FALSE, beta_ini = lastbeta, Hess = Hess,
                              N = Npast, lambda_inf = optlam,  etaloc=T, 
                              eta=eta1, method = "lasso" )
    
    indRL = which(Renewlasso[["beta_new"]] != 0)
    # Renewlasso[["beta_new"]][ indRL ]
    
    BetaRenewlasso[,k]  = Renewlasso[["beta_new"]]
    
    ## update candidate lambda list and beta list
    RLcandi[["lamcandi"]] = 10^seq( log10(optlam)+0.5, log10(optlam)-0.5, l=card) 
    eta1 = Renewlasso[["eta"]]
    ttl = matrix(0, p , card)
    
    for ( idx in 1: card){
      ttl[,idx] = Renew.online(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second,
                               warm = FALSE, beta_ini = lastbeta, Hess = Hess, etaloc=T, 
                               N = Npast, lambda_inf = RLcandi[["lamcandi"]][idx],
                               eta=eta1, method = "lasso" )[["beta_new"]]
    }
    RLcandi[["estcandi"]] = ttl
    
    
    time2 = Sys.time()
    
    cat(k, "/", M, ", time: ", time2-time1 , "\r") 
  }
  
  return( list(BetaRenewlasso = BetaRenewlasso ) )
}


OSIM_fin = function( Xtrain, Ytrain, 
                     eta2 = 0.8, card=7, Con=0.8){ 
  
  M = length(Xtrain)
  p = dim(Xtrain[[1]])[2]
  Xlist = Xtrain
  Ylist = Ytrain 
  
  # Preliminary
  g_fun    = function(x) return( log( 1+ exp(x) ) ) 
  g_prime  = function(x) return( exp(x)/(1+exp(x)) ) 
  g_second = function(x) return( exp(x)/(1+exp(x))/(1+exp(x)) )  
  
  ### first batch 
  Nnew = dim(Xlist[[1]])[1];
  Xb = Xlist[[1]]; Yb = Ylist[[1]]
  n = dim(Xlist[[1]])[1]
  
  
  ##### 5.OSIM #####
  # choose first-batch estimation via hBIC
  lamlist = 10^seq(-2, -0.5, l=card); 
  IClist=rep(0,card ); TempBeta = matrix(0, p, card) 
  
  part1 = 1:(n/2);   part2 = (n/2+1):n
  X1 = Xb[part1, ];     Y1 = Yb[part1]
  X2 = Xb[part2, ];     Y2 = Yb[part2]
  
  for( lam in 1:card ){
    tempb = OSIM(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, warm = FALSE,
                 beta_ini1 = rep(0,p), Hess1 = NULL, etaloc=T, 
                 beta_ini2 = rep(0,p), Hess2 = NULL,
                 N = 0, lambda_inf = lamlist[lam], eta= eta2, roundcoef=20) #double gradient, half learn
    TempBeta[,lam] =  tempb$beta_ave 
    tempre1 = X1 %*% tempb[["beta_new1"]];  tempre2 = X2 %*% tempb[["beta_new2"]]
    loss1 = 2*( sum( log(1+ exp(tempre1) ) ) - t(Y1) %*% tempre1 )
    loss2 = 2*( sum( log(1+ exp(tempre2) ) ) - t(Y2) %*% tempre2 )
    
    IClist[lam] = log(loss1) + log(loss2) +
      Con* log(log(p)) * log(n/2)/(n/2)*
      ( sum(tempb[["beta_new1"]]!=0) + sum(tempb[["beta_new2"]]!=0) )
  }
  
  mylambda = lamlist[ which.min(IClist)[1] ]
  apply(TempBeta,2, function(x) sum(x!=0));  IClist; 
  # indiht = which(TempBeta[,which.min(IClist)[1]]!=0)

  
  # use the proper lambda for learning
  RenewOSIM = OSIM(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, warm = FALSE, 
                   beta_ini1 = rep(0,p), Hess1 = NULL,
                   beta_ini2 = rep(0,p), Hess2 = NULL, etaloc=T, 
                   N = 0, lambda_inf = mylambda, eta=eta2, roundcoef=20) 
  
  
  
  ##### subsequent batch #####
  # estimator collection
  BetaOsim = matrix(0, p, M)
  
  BetaOsim[,1]       = RenewOSIM[["beta_ave"]]
  
  for( k in 2:M){
    Xb = Xlist[[k]]; Yb = Ylist[[k]]
    n = dim(Xlist[[k]])[1]
    
    Npast = Nnew;         # the cumulative sample size BEFORE this batch learning
    Nnew = Npast + n      # the cumulative sample size AFTER  this batch learning
    
    time1 = Sys.time()
    
    ##### 5.OSIM #####
    Hess1 = RenewOSIM[["Hess_new1"]];     Hess2 = RenewOSIM[["Hess_new2"]]; 
    lastbeta1 = RenewOSIM[["beta_new1"]]; lastbeta2 = RenewOSIM[["beta_new2"]]; 
    eta2 = RenewOSIM[["eta"]]
    mylambda = RenewOSIM[["laminf"]]*sqrt(Npast/Nnew)
    # adaptive tuning with hBIC
    lamlist = 10^seq( log10(mylambda)-1, log10(mylambda)+1, l=card)
    IClist=rep(0,card ); TempBeta = matrix(0, p, card) 
    
    part1 = 1:(n/2);  part2 = (n/2+1):n
    X1 = Xb[part1, ];     Y1 = Yb[part1]
    X2 = Xb[part2, ];     Y2 = Yb[part2]
    
    for( lam in 1:card ){
      tempb = OSIM(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, warm = FALSE,
                   beta_ini1 = lastbeta1,  Hess1 = Hess1,
                   beta_ini2 = lastbeta2,  Hess2 = Hess2,  etaloc=T, 
                   N = Npast, lambda_inf = lamlist[lam], 
                   eta=eta2, method = "lasso", roundcoef=20 ) #double gradient, half learn
      TempBeta[,lam] =  tempb$beta_ave
      b1 = tempb[["beta_new1"]];   tempre1 = X1 %*% b1
      b2 = tempb[["beta_new2"]];   tempre2 = X2 %*% b2
      loss1 = 2*( sum( log(1+ exp(tempre1) ) ) - t(Y1) %*% tempre1 ) +
        0.5* t(b1-lastbeta2) %*% Hess1 %*% (b1-lastbeta2)
      loss2 = 2*( sum( log(1+ exp(tempre2) ) ) - t(Y2) %*% tempre2 ) +
        0.5* t(b2-lastbeta1) %*% Hess2 %*% (b2-lastbeta1)
      
      IClist[lam] = log(loss1) + log(loss2) +
        Con* log(log(p)) * log(Nnew/2)/(Nnew/2)*
        ( sum(tempb[["beta_new1"]]!=0) + sum(tempb[["beta_new2"]]!=0) )
      # IClist[lam] = loss1 + loss2 + Con/5*max( ( sum(b1!=0) + sum(b2!=0) ), 2)*log(p)
    }
    mylambda = lamlist[ which.min(IClist)[1] ]
    apply(TempBeta,2, function(x) sum(x!=0));  IClist; 
    indiht = which(TempBeta[,which.min(IClist)[1]]!=0)
    # TempBeta[indiht, which.min(IClist)[1] ]
    

    RenewOSIM = OSIM(X=Xb, Y=Yb, g_prime=g_prime, g_second=g_second, warm = FALSE, 
                     beta_ini1 = lastbeta1,  Hess1 = Hess1, etaloc=T, 
                     beta_ini2 = lastbeta2,  Hess2 = Hess2, roundcoef=20,
                     N = Npast, lambda_inf = mylambda, eta=eta2, method = "lasso")
    BetaOsim[,k]  = RenewOSIM[["beta_ave"]]
    
    
    time2 = Sys.time()
    
    cat(k, "/", M, ", time: ", time2-time1 , "\r") 
  }
  
  return( list(BetaOsim=BetaOsim) )
}


RADAR_fin = function( Xtrain, Ytrain, card=7 ){ 
  
  M = length(Xtrain)
  p = dim(Xtrain[[1]])[2]
  Xlist = Xtrain
  Ylist = Ytrain 
  
  # Preliminary
  g_fun    = function(x) return( log( 1+ exp(x) ) ) 
  g_prime  = function(x) return( exp(x)/(1+exp(x)) ) 
  g_second = function(x) return( exp(x)/(1+exp(x))/(1+exp(x)) )  
  
  ### first batch 
  Nnew = dim(Xlist[[1]])[1];
  Xb = Xlist[[1]]; Yb = Ylist[[1]]
  n = dim(Xlist[[1]])[1]
  
  
  ##### 6.RADAR #####
  # Rk, alphak used values provided in Han2026JASA
  lamlist = 10^seq(-4,-1,l=card); 
  LogitRADAR = RADAR( X=Xb, Y=Yb, g_fun=g_fun, g_prime=g_prime, 
                      betapast = rep(0,p), lastlambdalist = lamlist,
                      betapastlist = matrix(0, p, card),
                      mylambdalist = lamlist,
                      Rk=10, alphak=10 )

  
  ##### subsequent batch #####
  # estimator collection
  BetaRADAR = matrix(0, p, M)
  
  BetaRADAR[,1]      = LogitRADAR[["betahat"]]
  
  
  for( k in 2:M){
    Xb = Xlist[[k]]; Yb = Ylist[[k]]
    n = dim(Xlist[[k]])[1]
    
    Npast = Nnew;         # the cumulative sample size BEFORE this batch learning
    Nnew = Npast + n      # the cumulative sample size AFTER  this batch learning
    
    time1 = Sys.time()
    
    ##### 6.RADAR #####
    #Rk, alphak update guided by Han2026JASA
    Rk               = LogitRADAR[["usedRk"]]/sqrt(2)
    alphak           = LogitRADAR[["usedalphak"]]*2
    RadarLamPastList = LogitRADAR[["usedlambdalist"]]
    RadarLamNowList  = LogitRADAR[["usedlambdalist"]]*sqrt(Npast/Nnew)
    betapast         = LogitRADAR[["betahat"]]
    betapastlist     = LogitRADAR[["usedbetaseq"]]
    
    LogitRADAR = RADAR(X=Xb, Y=Yb, g_fun=g_fun, g_prime=g_prime, 
                       betapast = betapast, lastlambdalist = RadarLamPastList,
                       betapastlist = betapastlist,
                       mylambdalist = RadarLamNowList,
                       Rk=Rk, alphak=alphak )
    
    BetaRADAR[,k] = LogitRADAR[["betahat"]]
    
    time2 = Sys.time()
    
    cat(k, "/", M, ", time: ", time2-time1 , "\r") 
  }
  
  return( list(BetaRADAR=BetaRADAR ) )
}
