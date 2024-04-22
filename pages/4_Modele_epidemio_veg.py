#application streamlit
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D


st.title("Système SIS avec dynamique d'imitation et un trade off clairance-transmission")

st.header("Présentation du système")

st.latex(r'''
    \left  \{
    \begin{array}{r c l}
        \frac{di}{d \tau}   & = &  b(\gamma) (1 - i)  i - \gamma i - 1 \\
        \frac{d \gamma}{d \tau}   & = & a  \gamma [b^{'}(\gamma)  (1 - i + \Sigma i) - 1 ] \\
   \end{array}
   \right.
             ''' 
)



st.header("Paramètres")
col1, col2 = st.columns(2)
  
with col1: 
    st.write("Temps de simulation")
    tmax = st.slider("Temps de simulation",1,1000)
    

    st.write("Paramètres de vitesse")
    sig =st.slider("Taux d'apprentissage",min_value = 0.0, max_value = 10.0,step = 0.1)
    
    
    A=st.slider("Variance de la clairance",min_value = 0.1, max_value = 10.0,step = 0.1)
    
N = 1

with col2: 
    st.write("Paramètres d'intérets")
    pay = st.slider("Rapport du payement des coopérateurs sur celui des défecteurs",min_value = 0.0, max_value = 1.0,step = 0.01)
    
    c = st.slider("Capacité d'infection constante",min_value = 0.0, max_value = 10.0,step = 0.1)
    k= st.slider("Paramètre de forme",min_value = 0.10, max_value = 1.0,step = 0.01)
    rho0 = st.slider("rho 0",min_value = 0.0, max_value = 10.0,step = 0.01)
    rho1 = st.slider("Rho 1",min_value = 0.0, max_value = 10.0,step = 0.01)


pas = 0.01
nbr_pas = int(tmax/pas)

###########################################Choix du trade offs

   

trade_choix = st.selectbox("Choix du trade-off",["cx^k","(x*c)/(k+x)"])
if trade_choix == "cx^k":
    def beta(x,c, k ):
        return(c*x**k)
    def beta2(x,c, k ):
        return(c*k*x**(k-1))
    st.subheader("Forme du trade off")
    droite1 = np.zeros(20)
    droite2 = np.zeros(20)
    droite3 = np.zeros(20)
    for y in range(20):
        droite1[y] = y
        droite2[y] = y+1
        droite3[y]=beta(y,c,k)
    figtrade, ax1 = plt.subplots()
    ax1.plot(range(20),droite1,"red")
    ax1.plot(range(20),droite2,"black")
    ax1.plot(range(20),droite3,"purple")
    ax1.set_xlabel('Clairance')
    ax1.set_ylabel('Transmission')
        

if trade_choix == "(x*c)/(k+x)":
    def beta(x,c, k ):
        return((x*c)/(k+x))
    def beta2(x,c, k ):
        return((c*k)/(k+x)**2)
    st.subheader("Forme du trade off")
    droite1 = np.zeros(20)
    droite2 = np.zeros(20)
    droite3 = np.zeros(20)
    for y in range(20):
        droite1[y] = y
        droite2[y] = y+1
        droite3[y]=beta(y,c,k)
    figtrade, ax1 = plt.subplots()
    ax1.plot(range(20),droite1,"red")
    ax1.plot(range(20),droite2,"black")
    ax1.plot(range(20),droite3,"purple")
    ax1.set_xlabel('Clairance')
    ax1.set_ylabel('Transmission')


    st.pyplot(figtrade)


################################################# Coeur du modèle

#Ancienne version non fonctionelle
def model(Y0, t ,sig,rho0,rho1,pay,c,k,A,N) :
    #I , gamma, x = Y0
    I =Y0[0]
    gamma =Y0[1]
    x =Y0[2]


    dI = beta(gamma ,c , k) * I * (1 - I) - gamma*I*(1+rho0*x) - I
    dgamma = A  *gamma*(beta2(gamma,c, k) *(1-I)- ( 1+rho0*x) )
    dx =  sig*x * (1-x)*( x*I - pay)
    return(dI,dgamma,dx)


def cooperators(i , gamma,x ,parms = [sig,rho0,rho1,pay,c,k,A,N] ):
    sig,rho0,rho1,pay,c,k,A,N = parms
    dx =  sig*x * (1-x)*(x*i - pay)
    return(dx)

def clairance(i , gamma,x, parms = [sig,rho0,rho1,pay,c,k,A,N]):
    sig,rho0,rho1,pay,c,k,A,N = parms
    
    dgamma = A  *gamma* (beta2(gamma,c, k) * (1 - i) - rho0 -rho1*x )    
    return(dgamma)


#######Runge kunta d'ordre 4
def runge_kunta_4(pas, I , gamma,x,parms = [sig,rho0,rho1,pay,c,k,A,N],name = None):
    sig,rho0,rho1,pay,c,k,A,N = parms
    
    if name == "gamma":
        k1 = clairance(I , gamma,x,parms = [sig,rho0,rho1,pay,c,k,A,N])
        k2 = clairance(I ,gamma + k1*pas/2 ,x, parms = [sig,rho0,rho1,pay,c,k,A,N])
        k3 = clairance(I ,gamma + k2*pas/2 ,x  ,parms = [sig,rho0,rho1,pay,c,k,A,N])
        k4 = clairance(I ,gamma + k3*pas ,x  ,parms = [sig,rho0,rho1,pay,c,k,A,N] )
        return(gamma + pas*(k1 + 2*k2 + 2*k3 +k4)/6) 
        
    if name == "coop":
        k1 = cooperators(I , gamma,x,parms = [sig,rho0,rho1,pay,c,k,A,N])
        k2 = cooperators(I ,gamma,x + k1*pas/2 , parms = [sig,rho0,rho1,pay,c,k,A,N] )
        k3 = cooperators(I ,gamma,x + k2*pas/2  ,parms = [sig,rho0,rho1,pay,c,k,A,N])
        k4 = cooperators(I ,gamma,x + k3*pas   ,parms = [sig,rho0,rho1,pay,c,k,A,N] )
        return(x + pas*(k1 + 2*k2 + 2*k3 +k4)/6) 


def better_ode( tmax, pas ,Y0,parms):
    di ,dgamma ,dx =Y0
    sig,rho0,rho1,pay,c,k,A,N = parms

    tab= np.zeros((nbr_pas,3))
    
    t = np.linspace(0,tmax,nbr_pas)
    
    for y in range(len(t)):
        i = di
        gamma = dgamma
        x = dx
        tab[y,0] = i
        tab[y,1] = gamma
        tab[y,2] = x
        #Evolution des compartiments
        
        dgamma = runge_kunta_4(pas, i , gamma,x,parms = [sig,rho0,rho1,pay,c,k,A,N], name = "gamma" )
        dx = runge_kunta_4(pas, i , gamma,x,parms = [sig,rho0,rho1,pay,c,k,A,N], name = "coop" )
        di = (i + pas * beta(gamma ,c , k)* (1 - x) * i)/(1 + beta(gamma ,c , k)*(1 - x )*i*pas +(gamma+1)*pas )
    
    return(tab)


        


#valeurs de départ

st.write("Valeurs initiales")
col21,col22,col23 = st.columns(3)
with col21:
    i0 = st.slider("Prévalence initiale",min_value = 0.01,max_value = 1.00, step = 0.01)
with col22:
    c0 = st.slider("Clairance initiale",1,100)
with col23:
    x0 = st.slider("Coopérateurs",min_value = 0.01,max_value = 1.00, step = 0.01)


###########################PLOT2D
st.subheader("Dynamiques des 3 compartiments en fonction du temps")
# sol = solve_ivp(model, y0 = [i0 , c0,x0], t_span = (0,tmax),args = (sig,rho0,rho1,pay,c,k,A,N),method="RK45",dense_output=True)
# sol = sol.y

temps = np.linspace(0,tmax,nbr_pas)
sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,rho0,rho1,pay,c,k,A,N))

#sol = better_ode( tmax , pas ,Y0 = [i0 , c0,x0],parms =[sig,rho0,rho1,pay,c,k,A,N])
# (Y0, tmax, pas ,parms)

temps = np.linspace(0,tmax,nbr_pas)

fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(temps,sol[:,0],"red")
ax1.plot(temps,sol[:,2],"black")
ax2.plot(temps,sol[:,1],"purple")


ax1.set_xlabel('Temps')
ax1.set_ylabel('Prévalence', color='red')
ax2.set_ylabel('Clairance', color='purple')

st.pyplot(fig1)
############################## PLOT 3D


repet = st.slider("Nombre de condition initial",0,100)

fig2 = plt.figure()
#ax = fig2.gca(projection='3d')
ax = plt.axes(projection='3d')
for i in range(repet):
    
    i0 = float(np.random.uniform(0.0001,0.99,1) )
    c0= float(np.random.uniform(0.01,100,1))
    x0 = float(np.random.uniform(0.001,0.9999,1))

    # sol = solve_ivp(model, y0 = [i0 , c0,x0], t_span = (0,tmax),args = (sig,rho0,rho1,pay,c,k,A,N),method="RK45",dense_output=True)
    # sol = sol.y
    # sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,rho0,rho1,pay,c,k,A,N))
    
    #sol = better_ode( tmax , pas ,Y0 = [i0 , c0,x0],parms =[sig,rho0,rho1,pay,c,k,A,N])
    
    temps = np.linspace(0,tmax,nbr_pas)
    sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,rho0,rho1,pay,c,k,A,N))

    x = sol[:,0]
    z = sol[:,1]
    y = sol[:,2]
    
    ax.plot(x, y, z)

ax.legend()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,100)

ax.set_xlabel('Prévalence')
ax.set_ylabel('Coopérateurs')
ax.set_zlabel('Clairance')
st.pyplot(fig2)



