#projeto curso python
# escrever a rotina N-PLS em linguagem python
# projeto de aula de tradução do script de MATLAB de:
# Copyright (C) 1995-2006  Rasmus Bro & Claus Andersson
# Copenhagen University, DK-1958 Frederiksberg, Denmark, rb@life.ku.dk
# $ Version 1.02 $ Date July 1998 $ Not compiled $
# $ Version 1.03 $ Date 4. December 1998 $ Not compiled $ Cosmetic changes
# $ Version 1.04 $ Date 4. December 1999 $ Not compiled $ Cosmetic changes
# $ Version 1.05 $ Date July 2000 $ Not compiled $ error caused weights not to be normalized for four-way and higher
# $ Version 1.06 $ Date November 2000 $ Not compiled $ increase max it and decrease conv crit to better handle difficult data
# $ Version 2.00 $ May 2001 $ Changed to array notation $ RB $ Not compiled $
# $ Version 2.01 $ June 2001 $ Changed to handle new core in X $ RB $ Not compiled $
# $ Version 2.02 $ January 2002 $ Outputs all predictions (1 - LV components) $ RB $ Not compiled $
# $ Version 2.03 $ March 2004 $ Changed initialization of u $ RB $ Not compiled $
# $ Version 2.04 $ Jan 2005 $ Modified sign conventions of scores and loads $ RB $ Not compiled $
# $ Version 3.00 $ Aug 2007 $ Serious error in sign switch fixed $ RB $ Not compiled $

import numpy as np
import matplotlib as plt
def projetonpls(X,Y,Fac,show):
    '''Essa função calcula o modelo N-PLS para dados X de terceira ordem em correlação com Y'''
    #INPUT => X:tensor de variaveis independentes; Y: tensor de variaveis dependentes; Fac: numero de componentes do modelo
    #OUTPUT => Xfactors: lista que contem as componentes de X; Yfactors: lista que contem as componentes de Y; Core: tensor conector; B: coeficientes do modelo; Ypred: valores de Y previstos; ssx: variação explicada no espaço X
    maxit = 120
    DimX = X.shape
    X = np.reshape(X,DimX[1],np.prod(DimX[2:-1]))
    ordX = len(DimX)
    if ordX==2:#size(Y,2)==1
        ordX = 1
    DimY = Y.shape
    Y = np.reshape(Y,DimY[1],np.prod(DimY[2:-1]))
    ordY = len(DimY)
    if ordY==2:
        ordY = 1
    [ix,jx] = X.shape
    [iy,jy] = Y.shape
    missX=0
    missy=0
    MissingX = 0
    MissingY = 0
    d1=np.isnan(X)
    d2=np.isnan(Y)
    if True in d1 or True in d2:
        if True in d2:
            MissingX=1
        else:
            MissingX=0
        if True in d2:
            MissingY=1
        else:
            MissingY=0
        soma=0
        for item in d1:
            if item == True:
                soma= 1+soma
            else:
                soma=0
        missX=abs(1-soma)
        soma=0
        for item in d1:
            if item == True:
                soma= 1+soma
        else:
            soma=0
        missy=abs(1-soma);
    crit=1e-10
    B=np.zeros((Fac,Fac))
    T = np.array(())
    U = np.array(())
    Qkron = np.array(())
    if MissingX ==1:
        SSX = sum(sum(X(missX.ravel().nonzero())**2)))
    else:
        SSX = sum(sum(X**2)))
    if MissingY ==1:
        SSY = sum(sum(Y(missY.ravel().nonzero())**2)))
    else:
        SSY = sum(sum(Y**2)))
    ssx=np.array(())
    ssy=np.array(())
    Xres=X
    Yres=Y
    xmodel=zeros(size(X));
    Q=[];
    W=[];
    
    for item in range(1,Fac+1):
        if shape(Yres,2)==1:
            u = Yres
        else:
            [u] = pcanipals(Yres,1,0)#verificar essa funcao no matlab
        t=random.uniform(DimX[1],1)
        tgl=t+2
        it=0
        while (norm(t-tgl)/norm(t)) > crit and it < maxit:
            tgl=t
            it=it+1
        if MissingX == 0:
            [wloads,wkron] = Xtu(X,u,MissingX,missX,Jx,DimX,ordX)#verificar
            for intem in range(1,I):
                m = missX[item,:,]
                m = m.ravel().nonzero()
                t(i)=np.dot(X(i,m),wkron(m))/np.dot(np.transpose(wkron(m),wkron(m)))
    
        else:
            t=X*wkron
        cc = corrcoef([t u]);
        if (cc[1,0]<0):
            t = -t;
             wloads{1}=-wloads{1}#
        T=[T t];
       for item in range(1:ordX-1):
           if num_lv == 1
           W{i} = wloads{i};
       else:
         W{i} = [W{i} wloads{i}];
     
       U=[U u];
       for item in range(1:max(ordY-1,1)):
           if num_lv == 1:
               Q{i} = qloads{i}
           else:
               Q{i} = [Q{i} qloads{i}]
               
       Qkron = [Qkron qkron]
       if ordX>1:
           Xfac{1}=T;Xfac(2:ordX)=W
           Core{num_lv} = calcore(reshape(X,DimX),Xfac,np.array(()),0,1)
      else:
          Core{num_lv} = 1
          
      B(1:num_lv,num_lv)=np.dot(linalg.inv(np.dot((np.transposeT),T),np.dot((np.transposeT),U[:,,num_lv])))

      if ordX>2:
          Wkron = kron(W{end},W{end-1})
      else:
          Wkron = W{end};
      for item in range(ordX-3:-1:1):
          Wkron = kron(Wkron,W{i});
   
      if num_lv>1:
          xmodel=np.dot(T,reshape(Core{num_lv},num_lv,np.dot(num_lv**(ordX-1)),np.transpose(Wkron)))
      else:
          xmodel = np.dot(T,Core{num_lv},np.transpose(Wkron))
          
      ypred=np.dot(T,B(1:num_lv,1:num_lv),(reshape(Ycore{num_lv},num_lv,num_lv**(ordY-1))),np.transpose(Qkron))   
      Xres=X-xmodel; 
      Yres=Y-ypred;
      if MissingX:
          ssx=array([[ssx,sum(sum(Xres(nonzero(missX)))**2))]])
      else:
          ssx=array([[ssx,sum(sum(Xres**2))]])
      if MissingY:             
      ssy=array([[ssy,sum(sum((Y(nonzero(missy))-ypred(nonzero(missy)))**2))]])
      else:
          ssy=array([[ssy,sum(sum((Y-ypred)**2))]]);

      ypred = np.reshape(np.transpose(Ypred),[shape(Ypred,2) DimY])
      ypred = np.transpose(ypred,[2:ordY+1 1])#verificar como eh o permute
      ssx= [ [SSX(1);ssx] [0;100*(1-ssx/SSX(1))]];
    ssy= [ [SSy(1);ssy] [0;100*(1-ssy/SSy(1))]];

    for item in range(1:Fac):
          tab = str(format,ssq(item,:,));
          print(tab)
    Xfactors = []
    Xfactors[0]=T
    for item in range(1:ordX-1):
        Xfactors[item+1]=W[item]
    Yfactors = []
    Yfactors[0]=U
    a=ordY-1
    for intem in range(1:a.max):
        Yfactors[item+1]=Q[item]
R = outerm(W,0,1)#verificar função
    for iy in range(1:shape(Y,2)):
        if DimX.size == 2:
            dd = [DimX[1], 1]
        else:
            dd = DimX[1:-1]
      
        for i in range(1:Fac):
            sR = np.dot(R[:,1:i],B[1:i,1:i],diag(Q[0],[iy,1:i]))
        ssR = np.transpose(sum(np.transpose(sR),1))
        reg[iy,i] = reshape(ssR,dd)
        function [wloads,wkron] = Xtu(X,u,Missing,miss,J,DimX,ord) #verificar
        
#fazer o produto da transposta de x com u
if Missing
   for i in range(1:J):
       m = nonzero(miss(:,i));
      if np.dot((np.transpose(u[m]),u[m]))!=0
        ww=np.dot(np.transpose(X[m,i]),u[m])/dot(np.transpose(u[m]),u[m])
      else
        ww=np.dot(np.transpose(X[m,i]),u[m])
      
      if size(ww)==0:
         w[i]=0
      else:
         w[i]=ww

else:
   w=np.dot(np.transpose(X),u)

# modificar a dimenssão da matriz de dados w
if size(DimX)>2:
   w_reshaped=np.reshape(w,DimX[1],np.prod(DimX[3:size(DimX)]));
else:
   w_reshaped = w[:,]


if size(DimX)==2
   wloads[0] = w_reshaped/norm(w_reshaped)
   
elif size(DimX)==3 and False in isnan(w_reshaped):
   [w1,s,w2]=linalg.svd(w_reshaped)
   wloads[0]=w1[:,,0]
   wloads[1]=w2[:,,0]
else
   wloads=parafac(reshape(w_reshaped,DimX(2:length(DimX))),1,[0 2 0 0 NaN]) # verificar a necessidade disso
   for j in range(0:size(wloads))
      wloads[j] = wloads[j]/norm(wloads[j])
for i in range(0:size(wloads))
   sq = (wloads[i]**2)*sign(wloads[i])
   wloads[i] = np.dot(wloads[i],(sum(sq)<0))
if size(wloads)==1
   wkron = wloads[0]
else
   wkron = kron(wloads[-1],wloads[-2])
   for o = [ord-3:-1:1]#criar uma lista decrescente de valores de ord-3 ate 1
      wkron = kron(wkron,wloads[o])
  


  


        




      


                
        
