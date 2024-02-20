#application streamlit
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns



def beta(x,c, k ):
    return((x*c)/(k+x))

def beta2(x,c, k ):
    return((c*k)/(k+x)**2)


def model(Y0, t ,sig,pay,c,k,A,N) :
    I , gamma, x = Y0
    
    
    dI = (1-x)* beta(gamma,c, k) * I * (1- I) - gamma * I
    dgamma = A*gamma *(beta2(gamma,c, k) * (N - I) -1 )
    dx = sig * x * (1-x)*( I - pay)





################## PLOT 2D
    

tmax = 1000
temps = np.linspace(0, tmax,tmax*100)

#parameters vitesse
sig =0.1
A=1
N = 1


pay = 0.2
c = 4
k= 1

pay_val = np.linspace(0, 1,5)
k_val = np.linspace(0, 10,5)
c_val = np.linspace(0, 10,4)
#valeurs de départ
for pay in pay_val:
    for k in k_val:
        for c in c_val:

            i0 , c0,x0 = 0.1, 30,0.1

            sol2 = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,pay,c,k,A,N))

            sns.lineplot(x= temps,y=sol2[:,0],color = "red")
            sns.lineplot(x= temps,y=1 -sol2[:,0],color = "green")
            sns.lineplot(x= temps,y=sol2[:,2],color = "yellow")
            plt.title(c)
            plt.show()


############################## PLOT 3D
            



