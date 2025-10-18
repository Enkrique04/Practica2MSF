"""
Práctica 1: Diseño de controladores

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Ian Enrique Estrada Castillo
Número de control: 22211753
Correo institucional: l22211753@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('Signal.xlsx',header=None))
x0,t0,tend,dt,w,h = 0,0,10,1E-3,7,3.5
n= round((tend - t0)/dt) + 1 
t = np.linspace(t0, tend, n)
u = np.reshape(signal.resample(u,len(t)),-1)

def cardio (Z,C,R,L):
    num = [L*R,R*Z]
    den = [C*L*R*Z,L*R+L*Z,R*Z]
    sys=ctrl.tf(num,den)
    return sys
#Funcion de transferencia: Normotenso
Z,C,R,L = 0.033,1.5,0.95,0.01
sysnormo = cardio(Z,C,R,L)
print(f'Funcion de transferencia del normotenso: {sysnormo}')

#Funcion de transferencia: Hipotenso
Z,C,R,L = 0.02,0.25,0.6,0.005
syshipo = cardio(Z,C,R,L)
print(f'Funcion de transferencia del hipotenso: {syshipo}')

#Funcion de transferencia: Hipertenso
Z,C,R,L = 0.05,2.5,1.4,0.02
syshiper = cardio(Z,C,R,L)
print(f'Funcion de transferencia del hipertenso: {syshiper}')

_,Pp0 = ctrl.forced_response(sysnormo,t,u,x0)
_,Pp1 = ctrl.forced_response(syshipo,t,u,x0)
_,Pp2 = ctrl.forced_response(syshiper,t,u,x0)

fgl= plt.figure()
plt.plot(t,Pp0,'-',linewidth = 1, color =[0.902,0.224,0.274],label='Pp(t):Normotenso')
plt.plot(t,Pp1,'-',linewidth = 1, color =[0.114,0.208,0.341],label='Pp(t):Hipotenso')
plt.plot(t,Pp2,'-',linewidth = 1, color =[0.569,0.392,0.235],label='Pp(t):Hipertenso')
plt.grid(False) #Para poner una cuadricula en la grafica
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t)[V]')
plt.ylabel('t[s]')
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'center', ncol = 3)
plt.show()
fgl.set_size_inches(w,h)
fgl.tight_layout()
fgl.savefig('Sistema Cardiovascular python.png',dpi=600,bbox_inches='tight')
fgl.savefig('Sistema Cardiovascular python.pdf')

def controlador (kP,kI,sys):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    X = ctrl.series(PI, sys)
    sysPI = ctrl.feedback(X,1,sign=-1)
    return sysPI

hipoPI = controlador (1.000340631088294622,404.434384377147,syshipo)
hiperPI = controlador(10,103.569197822789,syshiper) 

_,Pp3 = ctrl.forced_response(hipoPI,t,Pp0,x0)
_,Pp4 = ctrl.forced_response(hiperPI,t,Pp0,x0)

fgl= plt.figure()
plt.plot(t,Pp0,'-',linewidth = 1, color =[0.902,0.224,0.274],label='Pp(t):Normotenso')
plt.plot(t,Pp3,':',linewidth = 1, color =[0.271,0.482,0.616],label='Pp(t):HipotensoPI')
plt.plot(t,Pp2,'--',linewidth = 1, color =[0.114,0.208,0.341],label='Pp(t):Hipertenso')
plt.grid(False) #Para poner una cuadricula en la grafica
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t)[V]')
plt.ylabel('t[s]')
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'center', ncol = 3)
plt.show()
fgl.set_size_inches(w,h)
fgl.tight_layout()
fgl.savefig('Sistema Cardiovascular python HipoPI.png',dpi=600,bbox_inches='tight')
fgl.savefig('Sistema Cardiovascular python HipoPI.pdf')
"==========================================================================================="
fgl= plt.figure()
plt.plot(t,Pp0,'-',linewidth = 1, color =[0.902,0.224,0.274],label='Pp(t):Normotenso')
plt.plot(t,Pp1,':',linewidth = 1, color =[0.271,0.482,0.616],label='Pp(t):Hipotenso')
plt.plot(t,Pp4,'--',linewidth = 1, color =[0.114,0.208,0.341],label='Pp(t):HipertensoPI')
plt.grid(False) #Para poner una cuadricula en la grafica
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t)[V]')
plt.ylabel('t[s]')
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'center', ncol = 3)
plt.show()
fgl.set_size_inches(w,h)
fgl.tight_layout()
fgl.savefig('Sistema Cardiovascular python HiperPI.png',dpi=600,bbox_inches='tight')
fgl.savefig('Sistema Cardiovascular python HiperPI.pdf')