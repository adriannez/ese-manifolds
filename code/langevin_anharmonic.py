import numpy as np
import matplotlib.pyplot as plt
import sdeint
import scipy.special as fcns
import collections




#Define constants
T = 20.0 #Temp
gamma = 10.0 #Viscosity
k_0 = 1.0 #Initial Spring constant
k_c = 30.0 #Final spring constant
noise = np.sqrt(2*gamma*T)

num_sims = 100
t_i = 0.0
t_f = 30.0
delta_t = 0.1
N = int((t_f-t_i)/delta_t+1)

t_c = 15.0
ramping_f = 10



def ramp(z):
    "smooth ramp from 0 to 1"
    return 0.5*(1.0 + fcns.erf(z))

def dramp(z):
    "derivative of ramp"
    return np.exp(-z**2)/np.sqrt(np.pi)

def k_s(t):
    "value of k as a function of time"
    return k_0 + k_c*ramp(ramping_f*(t-t_c))


def dk_s(t):
    return ramping_f * dramp(ramping_f*(t-t_c))*(k_0)

def r(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)


def force(x,t):
    return x*gamma*dk_s(t) / (4.0 * k_s(t))

#x is (6,0) array:
    #   x[0] = x
    #   x[1] = y
    #   x[2] = z
    #   x[3] = v_x
    #   x[4] = v_y
    #   x[5] = v_z

B = np.diag([0, 0, 0, noise, noise, noise]) # diagonal, so independent driving Wiener processes

tspan = np.linspace(t_i, t_f, N);

def f(x, t):
    return np.array([1.0*x[3], 1.0*x[4],  1.0*x[5], -k_s(t)*r(x[0],x[1],x[2])**2*x[0] -gamma*x[3], -k_s(t)*r(x[0],x[1],x[2])**2*x[1] -gamma*x[4], -k_s(t)*r(x[0],x[1],x[2])**2*x[2] -gamma*x[5]])

def f_ese(x, t):
    return np.array([1.0*x[3], 1.0*x[4],  1.0*x[5], force(x[0],t)-k_s(t)*r(x[0],x[1],x[2])**2*x[0] -gamma*x[3], force(x[1],t)-k_s(t)*r(x[0],x[1],x[2])**2*x[1] -gamma*x[4], force(x[2],t)-k_s(t)*r(x[0],x[1],x[2])**2*x[2] -gamma*x[5]])

def G(x, t):
    return B

result = np.zeros((num_sims,N,6));
result_ese = np.zeros((num_sims,N,6));


for i in range(0,num_sims-1):
    x0 = [np.random.randn(), np.random.randn(),np.random.randn(),np.random.randn(),np.random.randn(),np.random.randn()]
    result[i] = sdeint.itoint(f, G, x0, tspan)
    result_ese[i] = sdeint.itoint(f_ese, G, x0, tspan)
    print(i)

print('Done with integrtion')



num_bins = 20;
a = np.ceil(abs(max(np.ndarray.flatten(result[:][:][0]), key=abs)))/3;
step_size = 2*a/num_bins;


def counts(iterable, low, high, bins):
    step = (high - low + 0.0) / bins
    dist = collections.Counter((float(x) - low) // step for x in iterable)
    return [dist[b] for b in range(bins)]


freq = np.zeros((num_bins,N));
freq_eq = np.zeros((num_bins,N));
freq_ese = np.zeros((num_bins,N));
rel_freq = np.zeros((num_bins,N));
mean_count = np.zeros((num_bins,1))
var_count = np.zeros((num_bins,1))
var = np.zeros((N,1))
var_ese = np.zeros((N,1))
var_eq = np.zeros((N,1))

print(np.shape(var))


binspan = np.linspace(-a, a, num_bins)

for i in range(0,N-1):
    for j in range(0,num_bins-1):
        k = k_s(tspan[i])
        result_vec = np.ndarray.flatten(result[:,i,0])
        freq[j,i] = 1./num_sims*counts(result_vec,-a+step_size*j,-a+step_size*(j+1),1)[0]
        result_ese_vec = np.ndarray.flatten(result_ese[:,i,0])
        freq_ese[j,i] = 1./num_sims*counts(result_ese_vec,-a+step_size*j,-a+step_size*(j+1),1)[0]
        freq_eq[j,i] = np.exp(-0.25*k*binspan[j]**4/T)
        mean_count[j] = freq_eq[j,i]*(binspan[j])
        var_count[j] = freq_eq[j,i]*(binspan[j]**2)
    freq_eq[:,i] /= np.sum(freq_eq[:,i])


    #var[i] = (sum(np.square(result_vec))-sum(result_vec)**2)/num_sims
    var[i] = sum(np.square(result_vec - np.mean(result_vec))) / num_sims
    #var_ese[i] = (sum(np.square(result_ese_vec))-sum(result_ese_vec)**2)/num_sims
    var_ese[i] = sum(np.square(result_ese_vec - np.mean(result_ese_vec))) / num_sims


    var_eq[i] = (sum(var_count)-sum(mean_count)**2)/num_sims



#k = np.zeros((N))
#for i in range(0,N-1):
#    k[i] = k_s(tspan[i])
#    for j in range(0,num_bins-1):
#        rel_freq[j,i] = freq[j,i] - freq_eq[j,i]


print('Only plotting left')


fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.contourf(tspan,binspan,freq)
ax2.contourf(tspan,binspan,freq_ese)
ax3.contourf(tspan,binspan,freq_eq)
plt.show()

fig2, (ax21, ax22, ax23) = plt.subplots(3)
ax21.plot(tspan,var)
ax22.plot(tspan,var_ese)
ax23.plot(tspan,var_eq)
plt.show()
