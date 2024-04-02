#application streamlit
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D


st.title("Système SIS avec dynamique d'imitation et un trade off virulence-transmission")

st.header("Présentation du système")
modele = st.radio("Hypothèse du modèle",["Population total constante","Population variable"])
if modele =="Population variable":
    st.latex(r'''
        \left  \{
        \begin{array}{r c l}
            \frac{d S}{d t}   & = & B - \beta(\gamma)SI - \mu S \\ 
            \frac{dI}{d t}   & = &  \beta(\gamma)SI (\alpha + \mu) I \\
            \frac{d \alpha}{d t}   & = & a  \alpha [b^{'}(\alpha)  (S + \sigma I) - 1 ] \\
            \frac{d x}{d t}   & = & \sigma x (1 -x) (\alpha I - \kappa)
    \end{array}
    \right.
                ''' 
    )

if modele =="Population total constante":
    st.latex(r'''
        \left  \{
        \begin{array}{r c l}
            \frac{dI}{d t}   & = &  \beta(\gamma)(N - I)I (\alpha + \mu) I \\
            \frac{d \alpha}{d t}   & = & a  \alpha [b^{'}(\alpha)  (N - I + \sigma I) - 1 ] \\
            \frac{d x}{d t}   & = & \sigma x (1 -x) (\alpha I - \kappa)
    \end{array}
    \right.
                ''' 
    )    


########################################################################## virulence et prévalence
    
virpre = st.radio("Forme de la virulence/prévalence",["forme 1","forme 2","forme 3","forme 4"])

if virpre == "forme 1":
    st.latex(r'''\alpha^{p} I^{q}''')
    def virulenceprevalence(alpha, i ,p,q):
        return(alpha**p *i**q)
if virpre == "forme 2":
    st.latex(r'''\alpha^{p} I^{1-p}''')
    def virulenceprevalence(alpha, i ,p,q):
        return(alpha**p *i**(1 - p))
if virpre == "forme 3":
    st.latex(r'''\frac{\alpha}{\alpha +\mu}^{p} \frac{I}{N}^{q}''')
    def virulenceprevalence(alpha, i ,p,q):
        return((alpha/(alpha+ 1))**p *(i/N)**q) 
if virpre == "forme 4":
    st.latex(r'''\frac{\alpha} \frac{I}{N}''')
    def virulenceprevalence(alpha, i ,p,q):
        return(alpha *(i/N) ) 


##############################################################################PARAMETRES






st.header("Paramètres")
col1, col2 = st.columns(2)
  
with col1: 
    st.write("Temps de simulation")
    tmax = st.slider("Temps de simulation",1,1000)
    

    st.write("Paramètres de vitesse")
    sig =st.slider("Taux d'apprentissage",min_value = 0.0, max_value = 10.0,step = 0.1)
    
    
    A=st.slider("Variance de la virulence",min_value = 0.1, max_value = 10.0,step = 0.1)
    
N = 1

with col2: 
    st.write("Paramètres d'intérets")
    pay = st.slider("Rapport du payement des coopérateurs sur celui des défecteurs",min_value = 0.1, max_value = 10.0,step = 0.01)
    
    c = st.slider("Capacité d'infection constante",min_value = 0.1, max_value = 10.0,step = 0.1)
    k= st.slider("Paramètre de forme",min_value = 0.1, max_value = 1.0,step = 0.01)
    supinfec = st.slider("Capacité de surinfection",min_value = 0.0, max_value = 1.0,step = 0.01)


p= st.slider("Paramètre de forme de la mortalité",min_value = 0.1, max_value = 1.0,step = 0.01)
q= st.slider("Paramètre de forme de la prévalence",min_value = 0.1, max_value = 1.0,step = 0.01)
pas = 0.01
nbr_pas = int(tmax/pas)


###########################################Choix du trade offs

affichage = st.toggle("Affichage du plot")   

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
    if affichage == True:
        figtrade, ax1 = plt.subplots()
        ax1.plot(range(20),droite1,"red")
        ax1.plot(range(20),droite2,"black")
        ax1.plot(range(20),droite3,"purple")
        ax1.set_xlabel('virulence')
        ax1.set_ylabel('Transmission')
        st.pyplot(figtrade)
        

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
    if affichage == True:
        figtrade, ax1 = plt.subplots()
        ax1.plot(range(20),droite1,"red")
        ax1.plot(range(20),droite2,"black")
        ax1.plot(range(20),droite3,"purple")
        ax1.set_xlabel('virulence')
        ax1.set_ylabel('Transmission')
        st.pyplot(figtrade)


    


################################################# Coeur du modèle

# Ancienne version non fonctionelle
def model(Y0, t ,sig,supinfec,pay,c,k,A,p,q,N) :
    I , alpha, x = Y0
    I =Y0[0]
    alpha =Y0[1]
    x =Y0[2]


    dI = (1 - x) * beta(alpha ,c , k) * I * (1 - I) - alpha * I - I
    dalpha = A  *alpha* ((1-x)*beta2(alpha,c, k) * (N - I + I* supinfec) - 1 )  
    dx =  sig*x * (1-x)*(virulenceprevalence(alpha,i,p ,q) - pay)
    return(dI,dalpha,dx)


def cooperators(i , alpha,x ,parms = [sig,supinfec,pay,c,k,A,p,q,N] ):
    sig,supinfec,pay,c,k,A,p,q,N = parms
    dx =  sig*x * (1-x)*(virulenceprevalence(alpha,i,p ,q) - pay)
    return(dx) 

def virulence(i , alpha,x, parms = [sig,supinfec,pay,c,k,A,p,q,N]):
    sig,supinfec,pay,c,k,A,p,q,N = parms
    
    dalpha = A  *alpha* ((1-x)*beta2(alpha,c, k) * (N - i + i* supinfec) - 1 )    
    return(dalpha)


#######Runge kunta d'ordre 4
def runge_kunta_4(pas, I , alpha,x,parms = [sig,supinfec,pay,c,k,A,p,q,N],name = None):
    sig,supinfec,pay,c,k,A,p,q,N = parms
    
    if name == "alpha":
        k1 = virulence(I , alpha,x,parms = [sig,supinfec,pay,c,k,A,p,q,N])
        k2 = virulence(I ,alpha + k1*pas/2 ,x, parms = [sig,supinfec,pay,c,k,A,p,q,N])
        k3 = virulence(I ,alpha + k2*pas/2 ,x  ,parms = [sig,supinfec,pay,c,k,A,p,q,N])
        k4 = virulence(I ,alpha + k3*pas ,x  ,parms = [sig,supinfec,pay,c,k,A,p,q,N] )
        return(alpha + pas*(k1 + 2*k2 + 2*k3 +k4)/6) 
        
    if name == "coop":
        k1 = cooperators(I , alpha,x,parms = [sig,supinfec,pay,c,k,A,p,q,N])
        k2 = cooperators(I ,alpha,x + k1*pas/2 , parms = [sig,supinfec,pay,c,k,A,p,q,N] )
        k3 = cooperators(I ,alpha,x + k2*pas/2  ,parms = [sig,supinfec,pay,c,k,A,p,q,N])
        k4 = cooperators(I ,alpha,x + k3*pas   ,parms = [sig,supinfec,pay,c,k,A,p,q,N] )
        return(x + pas*(k1 + 2*k2 + 2*k3 +k4)/6) 




def better_ode( tmax, pas ,Y0,parms):
    di ,dalpha ,dx =Y0
    sig,supinfec,pay,c,k,A,p,q,N = parms

    tab= np.zeros((nbr_pas,3))
    
    t = np.linspace(0,tmax,nbr_pas)
    
    for y in range(len(t)):
        i = di
        alpha = dalpha
        x = dx
        tab[y,0] = i
        tab[y,1] = alpha
        tab[y,2] = x
        #Evolution des compartiments
        
        dalpha = runge_kunta_4(pas, i , alpha,x,parms = [sig,supinfec,pay,c,k,A,p,q,N], name = "alpha" )
        dx = runge_kunta_4(pas, i , alpha,x,parms = [sig,supinfec,pay,c,k,A,p,q,N], name = "coop" )
        di = (i + pas * beta(alpha ,c , k)* (1 - x) * i)/(1 + beta(alpha ,c , k)*(1 - x )*i*pas +(alpha+1)*pas )
    
    return(tab)


####################################################################Calcul        


#valeurs de départ

st.write("Valeurs initiales")
col21,col22,col23 = st.columns(3)
with col21:
    i0 = st.slider("Prévalence initiale",min_value = 0.00,max_value = 1.00, step = 0.01)
with col22:
    c0 = st.slider("virulence initiale",1,100)
with col23:
    x0 = st.slider("Coopérateurs",min_value = 0.01,max_value = 1.00, step = 0.01)


###########################PLOT2D
st.subheader("Dynamiques des 3 compartiments en fonction du temps")
# sol = solve_ivp(model, y0 = [i0 , c0,x0], t_span = (0,tmax),args = (sig,supinfec,pay,c,k,A,p,q,N),method="RK45",dense_output=True)
# sol = sol.y

# sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,supinfec,pay,c,k,A,p,q,N))

sol = better_ode( tmax , pas ,Y0 = [i0 , c0,x0],parms =[sig,supinfec,pay,c,k,A,p,q,N])
# (Y0, tmax, pas ,parms)

temps = np.linspace(0,tmax,nbr_pas)

fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(temps,sol[:,0],"red")
ax1.plot(temps,sol[:,2],"black")
ax2.plot(temps,sol[:,1],"purple")
ax2.set_ylim(-0.5,10)

ax1.set_xlabel('Temps')
ax1.set_ylabel('Prévalence', color='red')
ax2.set_ylabel('virulence', color='purple')

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

    # sol = solve_ivp(model, y0 = [i0 , c0,x0], t_span = (0,tmax),args = (sig,supinfec,pay,c,k,A,p,q,N),method="RK45",dense_output=True)
    # sol = sol.y
    #sol = odeint(model, y0 = [i0 , c0,x0], t=temps,args = (sig,supinfec,pay,c,k,A,p,q,N))
    #sol = solve_ivp(model,[0,tmax], y0 = [i0 , c0,x0], t=temps,args = (sig,supinfec,pay,c,k,A,p,q,N),method="RK23")
    sol = better_ode( tmax , pas ,Y0 = [i0 , c0,x0],parms =[sig,supinfec,pay,c,k,A,p,q,N])
    
    x = sol[:,0]
    z = sol[:,1]
    y = sol[:,2]
    
    ax.plot(x, y, z)

ax.legend()
ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.set_xlabel('Prévalence')
ax.set_ylabel('Coopérateurs')
ax.set_zlabel('virulence')
st.pyplot(fig2)



