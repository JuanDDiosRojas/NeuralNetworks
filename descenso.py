import numpy as np

def descenso_grad(f,X0,eta):
   
    #definimos las derivadas parciales de una función de R^n a R
    #si X=(x1,x2,...,xk,...xn), debemos indicar con k respecto a cuál variable derivar
    def partial(g,k,X):
        h = 1e-9
        Y = np.copy(X)
        X[k-1]  =X[k-1]+h
        dp = (g(X)-g(Y))/h
        return dp
    #Ahora definimos la función que nos dará el gradiente
    def grad(f,X):
        grd=[]
        for i in np.arange(0,len(X)):
            ai=partial(f,i+1,X)
            grd.append(ai)
        return grd
    #Ahora se hacen las iteraciones
    i=0
    while True:
        i=i+1
        X0=X0-eta*np.array(grad(f,X0))
        if np.linalg.norm(grad(f,X0))<10e-8 or i>40: break
    return X0