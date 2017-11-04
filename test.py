import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

#fuel remaining as a function of density
def function(x):
    return (((np.exp(-x*0.2))*(1000*x+1000))-1000)/x

#finding density at which tanker does not reach B
def find_root(guess_value):
    root=fsolve(function,guess_value)
    return root



def plot(function,range):

    #initialize a figure
    fig=plt.figure(figsize=(20,10))
    #add title
    fig.suptitle("Oil delivered versus Density",fontsize=18)
    #add a subplot
    s_plot=fig.add_subplot(111)
    #Latex equation
    s_plot.text(40, 600, r'  $y=\frac{((e^{-x*0.2})*(1000)*(1+x)-1000)}{x}$', fontsize=25)
    #Assumed value box
    s_plot.text(80, 600, 'Assumed values=>\n C=0.2 litre/km/kg \n T=1000kg \n L=1000 litre', style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    #x label
    s_plot.set_xlabel("Density of oil(kg/litre)",fontsize=18)
    #y_label
    s_plot.set_ylabel("Oil reached to point B(litre)",fontsize=18)
    #finding limiting density
    root=find_root(15)
    #highlighting limiting density
    plt.plot([root],[0],'o')
    #inserting arrow annotation
    s_plot.annotate("("+str(np.round(root[0],3))+",0)"+" If density of fuel increases further nothing will reach B", xy=(root, 0), xytext=(30, 100),
                arrowprops=dict(facecolor='black', shrink=0.05))
    #coluring the invalid region
    plt.axvspan(root[0],100,facecolor='y')
    #plotting
    plt.plot(range,function(range))
    #showing
    plt.show()



#plot(function,np.arange(0,100,1))


e=np.random.rand()
print(e)