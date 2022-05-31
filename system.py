import numpy as np
from scipy import integrate
import scipy.integrate  as  ode
import matplotlib.pyplot  as  plt
from animator import animate

M = np.array([[-1, 1], [1, -1]])

def system(t, v, r, beta, sgamma, cgamma, K, p, q,  d):
    """Function to return the system of differential equations for two patch, 2-strain system.
    t: time variable
    v: wrapper vector for all variables: Should have 14 entries
    r: vector for the growth rates
    beta: list with vector for virus reproduction rates
    sgamma: list with vectors of single infection clearances in the form [[gamma1_1_, gamma1_2], [gamma2_1, gamma2_2]] (two elements) 
    cgamma: list with vector of co-infection clearances (gamma_11 = [gamma1_11, gamma2_11])) (4 elements)
    k: list with vectors for altered susceptibilities (k_ij = [k1_ij, k2_ij]) (4 elements)
    p: list with vectors [p1_ij, p2_ij] where pk_ij is probability that host from patch k coinfected with (i, j) transmits strain j. (4 elements)
    q: list with vectors [q1_ij, q2_ij] where qk_ij is probability that host from patch k coinfected with (i, j) transmits strain i. (4 elements)
    d: diffusion
    The ordering of the (i, j)'s should always be 11, 12, 21, 22.
    """
    n = int(len(v)/7)  #There's always 7 infection classes: S, I1, I2, I11, I12, I21, I22
    S = v[:n]
    I1 = v[n:2*n]
    I2 = v[2*n:3*n]
    I11 = v[3*n:4*n]
    I12 = v[4*n:5*n]
    I21 = v[5*n:6*n]
    I22 = v[6*n:]
    J1 = I1 + q[0]*I11 + p[0]*I11 + q[1]*I12 + p[1]*I21
    J2 = I2 + q[2]*I21 + p[2]*I12 + q[3]*I22 + p[3]*I22
    eqS = r*(1 - S) + sgamma[0]*I1 + sgamma[1]*I2 + cgamma[0]*I11 + cgamma[1]*I12 + cgamma[2]*I21 + cgamma[3]*I22 - beta[0]*S*J1 - beta[1]*S*J2 + d*M@S
    eqI1 = beta[0]*J1*S - (r + sgamma[0])*I1 - beta[0]*K[0]*I1*J1 - beta[1]*K[1]*I1*J2 + d*M@I1
    eqI2 = beta[1]*J2*S - (r + sgamma[1])*I2 - beta[0]*K[2]*I2*J1 - beta[1]*K[3]*I2*J2
    eqI11 = beta[0]*K[0]*I1*J1 - (r + cgamma[0])*I11 + d*M@I11
    eqI12 = beta[1]*K[1]*I1*J2 - (r + cgamma[1])*I12 + d*M@I12
    eqI21 = beta[0]*K[2]*I2*J1 - (r + cgamma[2])*I21 + d*M@I21
    eqI22 = beta[1]*K[3]*I2*J2 - (r + cgamma[3])*I22 + d*M@I22

    eqs = []
    for i in range(n):
        eqs.append(eqS[i])
    for i in range(n):
        eqs.append(eqI1[i])
    for i in range(n):
        eqs.append(eqI2[i])
    for i in range(n):
        eqs.append(eqI11[i])
    for i in range(n):
        eqs.append(eqI12[i])
    for i in range(n):
        eqs.append(eqI21[i])
    for i in range(n):
        eqs.append(eqI22[i])


    return eqs 


def solve(system, t, v0, r, beta, sgamma, cgamma, K, p, q, d):
    return integrate.solve_ivp(system, t, v0, args = (r, beta, sgamma, cgamma, K, p, q, d), dense_output=True, method='BDF')


#Two patch parameters
epsilon = 12
t = np.linspace(0, 400, 54)
v0 = [70, 70, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1]
N = sum(v0)
r =np.array([1.2, 1.2])
neutralbeta = 1
neutralgamma = 1.3
neutralk = 1
beta = [neutralbeta*np.array([1 + epsilon*1.6, 1 + epsilon*1.8]), neutralbeta*np.array([1+epsilon*2, 1 + epsilon*1.9])]
sgamma = [neutralgamma*np.array([1 + epsilon*0.8, 1 + epsilon*0.7]), neutralgamma*np.array([1 + epsilon*0.7, 1 + epsilon*0.6])]
cgamma = [neutralgamma*np.array([1 + epsilon*0.8,1 + epsilon*0.6]), neutralgamma*np.array([1+ epsilon*0.5, 1 + epsilon*1.4]), neutralgamma*np.array([1 + epsilon*1.2, 1 + epsilon*0.6]), neutralgamma*np.array([1 + epsilon*1.4, 1 + epsilon*1.2])]
K = [neutralk*np.array([1+ epsilon*2,1 + epsilon*1.2]), neutralk*np.array([1 + epsilon*1.1, 1 + epsilon*0.8]), neutralk*np.array([1 + epsilon*1.4, 1 + epsilon*1.6]), neutralk*np.array([1 + epsilon*1.4, 1 + epsilon*1.3])]
p = [np.array([1, 1]), np.array([0.5+epsilon*0.7, 0.5+epsilon*0.8]), np.array([0.5+epsilon*0.3, 0.5+epsilon*0.2]), np.array([1, 1])]
q = [np.array([1,1]), np.array([0.5+epsilon*0.7, 0.5+epsilon*0.8]), np.array([0.5+epsilon*0.3, 0.5+epsilon*0.2]), np.array([1, 1])]
d = 0.4

#Three Patch Parameters
v03 = [70, 70, 70, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
N = sum(v0)
r3 =np.array([1, 1.2, 0.8])
beta3 = [np.array([1.1, 1.2, 1.5]), np.array([1.2, 1.3, 1.4])]
sgamma3 = [np.array([0.8, 0.7, 1]), np.array([0.7, 0.6, 1.1])]
cgamma3 = [np.array([1,0.6, 1.2]), np.array([1, 0.4, 1.5]), np.array([1.2, 0.6, 0.9]), np.array([1.4, 1.2, 1.2])]
K3 = [np.array([1,1.2, 1]), np.array([1.1, 0.8, 0.9]), np.array([1.4, 1.6, 1.2]), np.array([1.4, 1.3, 1.2])]
p3 = [np.array([1,1, 1]), np.array([0.7, 0.8, 0.5]), np.array([0.3, 0.2, 0.5]), np.array([1, 1, 1])]
q3 = [np.array([1,1, 1]), np.array([0.7, 0.8, 0.5]), np.array([0.3, 0.2, 0.5]), np.array([1, 1, 1])]



solution = solve(system, t, v0, r, beta, sgamma, cgamma, K, p, q, d)


labels = ['S in 1', 'S in 2', 'I1 in 1', 'I1 in 2','I2 in 1','I2 in 2', 'I11 in 1', 'I11 in 2', 'I12 in 1', 'I12 in 2', 'I21 in 1', 'I21 in 2', 'I22 in 1', 'I22 in 2']
for i in range(12):
    plt.plot(t, solution.y[i][:len(t)], label = labels[i])

plt.legend()
plt.show()


fig, ax = plt.subplots(1, 2, figsize = (10,4))
fig.subplots_adjust(wspace = 0.5)
order = np.array(['A', 'B'])
for i in range(7):
    ax[0].plot(t, solution.y[2*i][:len(t)], label = labels[2*i])
    ax[1].plot(t, solution.y[2*i + 1][:len(t)], label = labels[2*i+1])


ax[0].set_title('Patch 1 Dynamics', fontsize = 16)
ax[1].set_title('Patch 2 Dynamics', fontsize = 16)
plt.legend()
plt.show()


#Strain Frequencies
I1star =  np.array([solution.y[2][-1], solution.y[3][-1]])
I2star =  np.array([solution.y[4][-1], solution.y[5][-1]])
I11star = np.array([solution.y[6][-1], solution.y[7][-1]])
I12star = np.array([solution.y[8][-1], solution.y[9][-1]])
I21star = np.array([solution.y[10][-1], solution.y[11][-1]])
I22star = np.array([solution.y[12][-1], solution.y[13][-1]])
I = I1star + I2star + I11star + I12star + I21star + I22star

z1 = np.array([(I1star + I11star + 0.5*I21star + +0.5*I12star)[0]/I[0], (I1star + I11star + 0.5*I21star + 0.5*I12star)[1]/I[1]])
print(z1)
z2 = np.array([(I2star + I22star + 0.5*I21star + 0.5*I12star)[0]/I[0], (I2star + I22star + 0.5*I21star + 0.5*I12star)[1]/I[1]])
print(z2)
print(z2 + z1)


def analysis(system ,v0, r, beta, sgamma, cgamma, K, p, q, d, epsilon):
    """Function to perform the analysis of the model. Inputs are initial conditions and parameters of the model and an option to plot."""
    m = r + sgamma
    measures = dict()
    solution = solve(system, t, v0, r, beta, sgamma, cgamma, K, p, q, d)
    Sstar = np.array([solution.y[0][-1], solution.y[1][-1]])
    I1star =  np.array([solution.y[2][-1], solution.y[3][-1]])
    I2star =  np.array([solution.y[4][-1], solution.y[5][-1]])
    I11star = np.array([solution.y[6][-1], solution.y[7][-1]])
    I12star = np.array([solution.y[8][-1], solution.y[9][-1]])
    I21star = np.array([solution.y[10][-1], solution.y[11][-1]])
    I22star = np.array([solution.y[12][-1], solution.y[13][-1]])
    measures['I1*'] = I1star
    measures['I2*'] = I2star
    measures['I11*'] = I11star
    measures['I12*'] = I12star
    measures['I21*'] = I21star
    measures['I22*'] = I22star
    I = I1star + I2star
    T = I1star + I2star + I11star + I12star + I21star + I22star
    D = T - I
    measures['T'] = T
    measures['I'] = I
    measures['D'] = D
    detP = np.array([np.linalg.det([[2*T[0], I[0]], [D[0], T[0]]]), np.linalg.det([[2*T[1], I[1]], [D[1], T[1]]])])

    z1 = np.array([(I1star + I11star + 0.5*I21star + +0.5*I12star)[0]/T[0], (I1star + I11star + 0.5*I21star + 0.5*I12star)[1]/T[1]])
    measures['z1'] = z1

    #Implement invasion fitness lambda_i_j for each patch
    #These are still vectors (one entry for each patch)     #This bit is wrong: Need to implement the neutral system parameters
    # Theta1 = (2*beta*Sstar*T**2)/detP
    # Theta2 = sgamma*I*(I + T)/detP
    # Theta3 = sgamma*T*D/detP
    # Theta4 = 2*m*T*D/detP
    # Theta5 = beta*T*I*D/detP
    # Theta = Theta1 + Theta2 + Theta3 + Theta4 + Theta5
    # theta1 = Theta1/Theta
    # theta2 = Theta2/Theta
    # theta3 = Theta3/Theta
    # theta4 = Theta4/Theta
    # theta5 = Theta5/Theta
    # mu = I/D

    # lambda1_2 = theta1*(beta[0] -  beta[1]) + theta2*(sgamma[0] - sgamma[1]) + theta3*(-cgamma[1] - cgamma[2] + 2*cgamma[3]) + theta4*((q[1] - p[2])/epsilon) + theta5*(mu*(K[2] - K[1]) + K[2] - K[3])
    # lambda2_1 = theta1*(beta[1] - beta[0]) + theta2*(sgamma[1] - sgamma[0]) + theta3*(-cgamma[2] - cgamma[1] + 2*cgamma[0]) + theta4*((q[2] - p[1])/epsilon) + theta5*(mu*(K[1] - K[2]) + K[1] - K[0])
    # measures['lambda1_2'] = lambda1_2
    # measures['lambda2_1'] = lambda2_1

    return measures



ds = np.linspace(0, 500, 100)
zs1 = []
zs2 = []
for d in ds:
    s = analysis(system, v0, r, beta, sgamma, cgamma, K, p, q, d, epsilon)['z1']
    zs1.append(s[0])
    zs2.append(s[1])

# test = [analysis(system, v0, r, beta, sgamma, cgamma, K, p, q, d, 0.1)['lambda1_2'], analysis(system, v0, r, beta, sgamma, cgamma, K, p, q, d, 0.1)['lambda2_1']]
# print(test[0] - test[1])

animate(ds, zs1, zs2, ['d', 'z1'])

plt.show()


plt.plot(ds, zs1, label = 'z1 of first patch')
plt.plot(ds, zs2, label = 'z1 of second patch')
plt.title('Variation of z as function of diffusion')
plt.xlabel('d')
plt.ylabel('z1')

plt.legend()
plt.show()