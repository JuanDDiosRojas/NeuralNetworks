import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import mplcyberpunk
import tqdm
from functions import nth_derivative#, Param_dirich
plt.style.use('cyberpunk')

zi=0.0
zf=3.0
#t=torch.linspace(zi,zf,150).view(-1,1)
x0_i=0.2
x0_f=0.5

###################################################################################
nodos=50
ANN_param = nn.Sequential(nn.Linear(2, nodos), nn.Tanh(),
                           nn.Linear(nodos,nodos),
                    nn.Tanh(),nn.Linear(nodos,1))

ANN_param.load_state_dict(torch.load('L-CDM_param_dict_50'))
ANN_param.eval()

#param
def Param(T,net=ANN_param,ti=zi):
    out = net(T)
    b=1-torch.exp(ti-T[:,0])
    return T[:,1].view(-1,1) +b.view(-1,1)*out

#intervals for the grid
z_mesh = np.linspace(zi, zf, 100)
x0_mesh = np.linspace(x0_i, x0_f, 100)

z_param = torch.linspace(zi, zf, 100)
x0_param = torch.linspace(x0_i, x0_f, 100)
#grid
mesh=np.ones((100,100))

#evaluation of the grid
for i in range(100):
    for j in range(100):
        a=Param(torch.tensor([[z_param[i],x0_param[j]]]), net=ANN_param).detach().numpy()
        #b=sol_x([x0_mesh[j],1.0], [z_mesh[i]], 1.5, 3.0)
        b=x0_mesh[j] * (z_mesh[i]+1)**3
        
        mesh[i,j] =  abs(a-b)/abs(b) * 100.0

#density map
fig, ax = plt.subplots()

# Creamos la barra de densidad
pcolormesh = ax.pcolormesh(z_mesh, x0_mesh, mesh, cmap='inferno')

# Modificamos la función `colorbar` para que muestre el símbolo de porcentaje
colorbar = plt.colorbar(pcolormesh, format='%1.1f%%')
colorbar.ax.set_ylabel('percent error', size=13)
ax.set_xlabel('$z$', size=16)
ax.set_ylabel('$\Omega_{m,0}$', size=16)
ax.set_title('$\Lambda-CDM$ 50 nodes', size=16)
# Mostramos la gráfica
plt.show()
#plt.savefig()
fig.savefig('L-CDM_50.pdf')






