#Functions
def simulateLIF(dx,I,C,gL,EL,VT,W):
    '''Approximate LIF Neuron dynamics by Euler's method.

    Parameters
    ----------
    T, dT :number
       Total time being simulated (ms) and Euler time-step (ms)
       
    C,gL,EL,VT:number
        neuronal parameters
     
       
    I:1D NumPy array
        Input current (pA)
    
    W:1D NumPy array
        Synaptic Weight Strengths  - Connectivity Matrix
    
    Returns
    -------
    V,Isyn :  NumPy array (mV), NumPy array (pA)
        Approximation for membrane potential computed by Euler's method.
    '''
    
    V=EL*np.ones((np.shape(I)[0],2*np.shape(I)[1])) #initialising V
    stim=np.zeros((np.shape(I)[0],2*np.shape(I)[1])) #initialising synaptic stimulus

    Isyn=np.zeros((np.shape(I)[0],2*np.shape(I)[1])) #initialising synaptic current to zero
    dx=dt
    for n in range(0,np.shape(I)[1]-1):
        V[:,n+1] = V[:,n] + (dx/C)*(I[:,n]+Isyn[:,n] -gL*(V[:,n]-EL))
        check=np.where(V[:,n+1]>VT)[0]
        if np.size(check)>0:
            V[check,n+1]=EL
            #V[check,n]=1.8*VT #Artificial Spike
            stim[check,n+1:n+1+Nk] += ker
            Isyn[:,n+1:] = 1000*np.matmul(W,stim[:,n+1:])
    return V,Isyn 


C = 300
gL = 30
EL = -70
VT = 20
T = 5000   
dt = .1 
iter = np.int_(T/dt) #Number of timesteps

Numneurons = 4 
Iinp = np.zeros(1)
ton = 200
toff = 200
 
pulse = np.hstack((3200*np.ones(ton),0*np.ones(toff)))
Iinp1 = np.hstack([pulse, 0*np.ones((ton+toff)*2)])

Iinp2 = np.hstack([np.ones((ton+toff)), pulse])
Iinp2 = np.hstack([Iinp2, np.ones((ton+toff))])
Iinp3 = np.hstack([ 0*np.ones((ton+toff)*2),pulse])
Iinp4 = ([ 0*np.ones((ton+toff)*3)])
    
I1 = np.vstack((Iinp1,Iinp2))
I1 = np.vstack((I1,Iinp3))
I1 = np.vstack((I1,Iinp4))

I2 = np.vstack((Iinp1,Iinp3))
I2 = np.vstack((I2,Iinp2))
I2 = np.vstack((I2,Iinp4))

# Obtaining kernel
kerlen = 80
Nk = np.int_(kerlen/dt)
xker = np.linspace(0, kerlen,Nk , endpoint=True)
ker = np.array( np.exp(-xker/15)-np.exp(-xker/3.75) )


#Connectivity matrix, selectively estimated weights to elicit a spike when A->B->C only occurs
W=[[0,0,0,0],
   [0,0,0,0],
   [-5,0,0,0],
   [5,5,10,0],]

#W=np.random.randint(5, size=(Numneurons,Numneurons))
 
V1,I1s =simulateLIF(dt,I1,C,gL,EL,VT,W)
V2,I2s =simulateLIF(dt,I2,C,gL,EL,VT,W)

t=dt*(np.arange(np.shape(I1)[1]))

fig, axs = plt.subplots(1,2,figsize=(35,15))
plt.grid(True) 
#axs[0].plot(t ,Iinp/2700 ,'bo-', )
for n in range(0,Numneurons):
    axs[0].plot(t,2+2*n+(I1[n,0:np.shape(I1)[1]]-EL)/2700 , 'b-' )
    axs[1].plot(t,2+2*n+(I2[n,0:np.shape(I1)[1]]-EL)/2700, 'b-' )
    
    axs[0].plot(t,2.5+2*n+(V1[n,0:np.shape(I1)[1]]-EL)/(3*VT-EL) , 'r-' )
    axs[0].plot(t,8+2*n+(I1s[n,0:np.shape(I1)[1]])/(4000) , 'g-')
    axs[1].plot(t,2.5+2*n+(V2[n,0:np.shape(I1)[1]]-EL)/(3*VT-EL) , 'r-' )
    axs[1].plot(t  ,8+2*n+(I2s[n,0:np.shape(I1)[1]])/(4000) , 'g-')
axs[0].title.set_text("A->B->C")
axs[1].title.set_text("A->C->B")
axs[0].grid(True) 
axs[1].grid(True) 
axs[1].set(xlabel="t (ms)", ylabel="Probability")
plt.show(block = False)