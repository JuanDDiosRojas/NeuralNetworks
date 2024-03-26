import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
#import mplcyberpunk
import tqdm
from functions import nth_derivative#, Param_dirich
#plt.style.use('cyberpunk')

#########################################################

#Domain intervals

zi=0.0
zf=3.0
#t=torch.linspace(zi,zf,150).view(-1,1)
x0_i=0.2
x0_f=0.5

T=torch.cartesian_prod(torch.linspace(zi,zf,1000),
                       torch.linspace(x0_i,x0_f,40))

#random permutation of the training dataset T
T=T[torch.randperm(T.shape[0])]

#if avaliable, use cuda
if torch.cuda.is_available(): T.cuda()

#Neural network architecture

nodos=25
ANN = nn.Sequential(nn.Linear(2, nodos), nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    #nn.Tanh(), nn.Linear(nodos,nodos),
                    # nn.Tanh(), nn.Linear(nodos,nodos)
                    nn.Tanh(),nn.Linear(nodos,1))
print(ANN)

#if torch.cuda.is_available(): T.cuda()

#cost function
def cost(T):
    x=ANN(T)
    z0=torch.zeros_like(T[:,1]).view(-1,1)
    z0=torch.cat((z0, T[:,1].view(-1,1)), 1)
    #z0.requires_grad=True
    z=T[:,0].view(-1,1)
    Dx = nth_derivative(ANN,T,0,0,1)
    osc = Dx - (3*x / (1.0+z))
    omega_0 = ANN(z0) - T[:,1].view(-1,1)
    #omega_0.requires_grad=True
    return torch.mean(osc**2) + torch.mean(omega_0**2)

#Training loop
epochs=[3000,3000,1000]
tasas=[0.01,0.001,0.0005]
errores=[]
for k in range(len(epochs)):
    learning_rate=tasas[k]
    epocas=epochs[k]

    #optimizer=torch.optim.SGD(ANN.parameters(),lr=learning_rate,momentum=0.9)
    optimizer = torch.optim.Adam(ANN.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(epocas), desc="Training",  colour='cyan', ncols=100)
    for i in pbar:
        l=cost(T) #coste
        l.backward() #gradiente
        optimizer.step() #se actualizan los parámetros
        optimizer.zero_grad() #vacíamos el gradiente
        errores.append(float(l))
        pbar.set_postfix({'loss': l.item()})
plt.figure(figsize=(8, 6))
plt.plot(range(np.sum(epochs)),errores)
plt.xlabel('Épocas', size=18, color='pink')
plt.ylabel('$\mathcal{L}_\mathcal{F}$', size=18, color='pink')
plt.title('Entrenamiento 20 nodos 1000 $z$: enfoque costo', size=18, color='pink')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('LCDM_20nodos_1000z_cost.pdf', dpi=300)
# Leyenda
#plt.show()
# Guardar
#saving the trained model
torch.save(ANN.state_dict(),'LCDM_cost20_1000z')

































