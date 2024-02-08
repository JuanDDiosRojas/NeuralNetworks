
import torch
from scipy.integrate import odeint
import numpy as np
def nth_derivative(net, x, n):
    """Esta función está creada para regresar la derivada temporal de la función ann, asumiendo que 
    la primer columna de x es el tiempo siempre
    x=[t, param_1, param_2,...] 
    
    x debe ser un tensor: torch.Size([n, m])

    Para el buen funcionamiento: net(x)[0].shape = torch.Size([1])
    """
    x.requires_grad=True
    # Initialize the gradient tensor to 1.0, as the 0th derivative is the function itself.
    ann=net(x)
    grad_tensor = torch.ones(ann.size(), dtype=torch.float32, requires_grad=True)
    for _ in range(n):
        Dann=torch.autograd.grad(ann, x, grad_outputs=grad_tensor, create_graph=True)[0][:,0]
        Dann=Dann.reshape(ann.shape)
        ann=Dann
    return ann
####################################################################################
def oscilador(Y,t,delta,omega):
  x,dx=Y
  #queremos el arreglo [y,F]
  #y=dx
  #F nos la da la ecuación del oscilador
  return [dx,-delta*dx - omega**2 * x]

# #Condiciones del sistema:
# y0=[0,1] #[x0,y0]
# delta,omega = 0.8,5 #parametros libres
# t = np.linspace(0, 10, 100) #tiempo

#la función sol del sistema
def sol_x(y0, t, delta, omega):
   return odeint(oscilador, y0, t, args=(delta, omega))[:,0]

#############################################################################
# # Parámetros del oscilador
# m = 1.0  # Masa
# k = 10.0  # Constante de elasticidad
# c = 1.0  # Constante de amortiguamiento

# # Condiciones iniciales
# x0 = 1.0  # Posición inicial
# v0 = 0.0  # Velocidad inicial

# # Función derivada del oscilador
# def f(y, t):
#     x, v = y
#     return np.array([v, -c / m * v - k / m * x])

# # Tiempo de integración
# t_max = 10.0
# t = np.linspace(0.0, t_max, 1000)

# # Solución del oscilador
# y = odeint(f, [x0, v0], t)