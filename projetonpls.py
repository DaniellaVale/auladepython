#projeto curso python
# escrever a rotina N-PLS em linguagem python
import numpy as np

def projetonpls(X,Y,Fac,show):
    #X.argmin() 
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
      ypred = np.permute(ypred,[2:ordY+1 1])



      


                
        
