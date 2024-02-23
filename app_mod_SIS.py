#application streamlit
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D



st.title("Système SIS avec dynamique d'imitation et un trade off clairance-transmission")

st.header("Présentation du système")

st.write("On part du système SIS suivant:")
st.latex(r'''
    \left  \{
    \begin{array}{r c l}
      \frac{dS}{dt}  & = & B + \gamma I -\beta  S  I \\
      \frac{dI}{dt}   & = & \beta  S  I - \gamma I 
   \end{array}
   \right.
             ''' 
)
st.write("Que l'on réduit à une équation en posant N=S+I:")
st.latex(r'''
    \left  \{
    \begin{array}{r c l}
      \frac{dI}{dt}   & = & \beta  (N - I)  I - \gamma I 
   \end{array}
   \right.
             ''' 
)
st.write("On couple ce système avec avec une dynamique d'imitation et un trade off transmission-clairance:")
st.latex(r'''
    \left  \{
    \begin{array}{r c l}
        \frac{dI}{dt}   & = & (1 - \frac{X}{N})\beta(\gamma)  (N - I)  I - \gamma I \\
        \frac{d \gamma}{dt}   & = & A  \gamma [\beta^{'}(\gamma) (N - I) - 1 ] \\
        \frac{dX}{dt}   & = & \sigma  \frac{X}{N}  (1-\frac{X}{N})(r_D  \frac{I}{N} - r_C)
   \end{array}
   \right.
             ''' 
)
st.write("On addimentionne ce système en posant:")
st.latex(r'''
         i = \frac{I}{N}, c = \frac{\gamma}{\sigma}, \tau = t  \sigma, x = \frac{X}{N},\kappa = \frac{\pi_C}{\pi_D}, a = \frac{A}{\sigma}, b(\gamma)=\frac{N \beta(\gamma)}{\sigma}
         ''' 
)
st.write("Ce qui nous donne le système suivant:")
st.latex(r'''
    \left  \{
    \begin{array}{r c l}
        \frac{di}{d \tau}   & = & (1 - x ) b(c) (1 - i)  i - c i \\
        \frac{dc}{d \tau}   & = & a  c [b^{'}(c)  (1 - i) - 1 ] \\
        \frac{dx}{d \tau}   & = &  x  (1-x)( i - \kappa)
   \end{array}
   \right.
             ''' 
)


st.header("Paramètres")
col1, col2 = st.columns(2)
  
with col1: 
    st.write("Temps de simulation")
    tmax = st.slider("Temps de simulation",1,1000)
    temps = np.linspace(0, tmax,tmax*100)

    st.write("Paramètres de vitesse")
    sig =st.slider("Taux d'apprentissage",min_value = 0.0, max_value = 10.0,step = 0.1)
    A=st.slider("Variance de la clairance",min_value = 0.0, max_value = 10.0,step = 0.1)
    N = 1

with col2: 
    st.write("Paramètres d'intérets")
    pay = st.slider("Rapport du payement des coopérateurs sur celui des défecteurs",min_value = 0.0, max_value = 10.0,step = 0.1)
    c = st.slider("Capacité d'infection constante",min_value = 0.0, max_value = 10.0,step = 0.1)
    k= st.slider("Paramètre de forme",min_value = 0.0, max_value = 1.0,step = 0.01)




###########################################Choix du trade offs
trade_choix = st.selectbox("Choix du trade-off",["cx^k","(x*c)/(k+x)"])
   
    
if trade_choix == "cx^k":
    def beta(x,c, k ):
        return(c*x**k)
    def beta2(x,c, k ):
        return((c/k)*x**(k-1))
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




def model(Y0, t ,sig,pay,c,k,A,N) :
    I , gamma, x = Y0
    
    
    dI = (1 - x) * beta(gamma ,c , k) * I * (1 - I) - gamma * I
    dgamma = A  *gamma* (beta2(gamma,c, k) * (N - I) - 1 )
    dx =  sig*x * (1-x)*( I - pay)
    return(dI,dgamma,dx)
#valeurs de départ

st.write("Valeurs initiales")
col21,col22,col23 = st.columns(3)
with col21:
    i0 = st.slider("Prévalence initiale",min_value = 0.01,max_value = 1.00, step = 0.01)
with col22:
    c0 = st.slider("Clairance initiale",0,100)
with col23:
    x0 = st.slider("Coopérateurs",min_value = 0.01,max_value = 1.00, step = 0.01)


###########################PLOT2D
st.subheader("Dynamiques des 3 compartiments en fonction du temps")
sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,pay,c,k,A,N))


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
    
    sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,pay,c,k,A,N))
    x = sol[:,0]
    z = sol[:,1]
    y = sol[:,2]
    
    ax.plot(x, y, z)

ax.legend()
ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.set_xlabel('Prévalence')
ax.set_ylabel('Coopérateurs')
ax.set_zlabel('Clairance')
st.pyplot(fig2)



