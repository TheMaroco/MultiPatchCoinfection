import numpy as np
from scipy import integrate
import scipy.integrate  as  ode
import matplotlib.pyplot  as  plt
from animator import animate
from animation import Animate

M = np.array([[-1, 1], [1, -1]])

def replicator(tau, z, Theta, lambda1_2, lambda2_1, weight):
    """Function for the replicator equation using the summarized parameters."""
    #lambda1_2 = theta1(b2 - b1) + theta2(-nu2 + nu1) + theta3*(-u21 - u12 + u11) + theta4(omega2_21 - omega1_12) + theta5*(mu(alpha12 - alpha21) + alpha12 - alpha11)
    #lambda2_1 = theta1(b2 - b1) + theta2(-nu2 + nu1) + theta3*(-u21 - u12 + u11) + theta4(omega2_21 - omega1_12) + theta5*(mu(alpha12 - alpha21) + alpha12 - alpha11)
    Z = np.zeros((2,2))                 #This is going to be a matrix where collumns are patches and rows are strains.
    Z[0, 0] = z[0]
    Z[0, 1] = z[1]
    Z[1, 0] = z[2]
    Z[1, 1] = z[3]
    Z[1, 0] = 1 - Z[0, 0]
    
    Z[0, 1] = 1 - Z[1, 1]
    Lambdas = [np.array([[0, lambda1_2[0]],[lambda2_1[0], 0]]), np.array([[0, lambda1_2[1]],[lambda2_1[1], 0]])]


    eqlist = []
    #Order of equations is: z11, z12, z21, z22. So i is refering to the strain and j is refering to the patch.
    for i in range(2):
        for j in range(2):
            eqlist.append(Theta[j]*Z[i, j]*((Lambdas[j]@Z[:,j])[i] - np.transpose(Z[:, j])@Lambdas[j]@Z[:, j]) + (weight[j]-1)*(Z[i, (j+1)%2] - Z[i, j]))


    return eqlist

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
    return integrate.solve_ivp(system, t, v0, args = (r, beta, sgamma, cgamma, K, p, q, d), dense_output=True, method='BDF', rtol = 1e-13)


#Two patch parameters
epsilon = 0.1
t = np.linspace(0, 400, 100)
dt = t[1] - t[0]
tau = t*epsilon
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
d = epsilon

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




def analysis(system, v0, r, neutralbeta, beta, neutralgamma, sgamma, cgamma, neutralk,  K, p, q, d, epsilon):
    """Function to perform the analysis of the model. Inputs are initial conditions and parameters of the model and an option to plot."""
    m = r + neutralgamma
    measures = dict()

    solution = solve(system, t, v0, r, beta, sgamma, cgamma, K, p, q, d)
    measures['solution'] = solution

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
    T =I1star + I2star + I11star + I12star + I21star + I22star
    I = I1star + I2star
    D = T - I
    measures['T'] = T
    measures['I'] = I
    measures['D'] = D
    detP = np.array([np.linalg.det([[2*T[0], I[0]], [D[0], T[0]]]), np.linalg.det([[2*T[1], I[1]], [D[1], T[1]]])])
    z1 = np.array([(I1star + I11star + 0.5*I21star + +0.5*I12star)[0]/T[0], (I1star + I11star + 0.5*I21star + 0.5*I12star)[1]/T[1]])
    z2 = np.array([(I2star + I22star + 0.5*I21star + +0.5*I12star)[0]/T[0], (I2star + I22star + 0.5*I21star + 0.5*I12star)[1]/T[1]])
    measures['z1'] = z1
    measures['z2'] = z2

    #Implement invasion fitness lambda_i_j for each patch
    #These are still vectors (one entry for each patch)     
    Theta1 = (2*neutralbeta*Sstar*T**2)/detP
    Theta2 = neutralgamma*I*(I + T)/detP
    Theta3 = neutralgamma*T*D/detP
    Theta4 = 2*m*T*D/detP
    Theta5 = neutralbeta*T*I*D/detP
    Theta = Theta1 + Theta2 + Theta3 + Theta4 + Theta5
    theta1 = Theta1/Theta
    theta2 = Theta2/Theta
    theta3 = Theta3/Theta
    theta4 = Theta4/Theta
    theta5 = Theta5/Theta
    mu = I/D

    lambda1_2 = theta1*(beta[0] -  beta[1]) + theta2*(sgamma[0] - sgamma[1]) + theta3*(-cgamma[1] - cgamma[2] + 2*cgamma[3]) + theta4*((q[1] - p[2])/epsilon) + theta5*(mu*(K[2] - K[1]) + K[2] - K[3])
    lambda2_1 = theta1*(beta[1] - beta[0]) + theta2*(sgamma[1] - sgamma[0]) + theta3*(-cgamma[2] - cgamma[1] + 2*cgamma[0]) + theta4*((q[2] - p[1])/epsilon) + theta5*(mu*(K[1] - K[2]) + K[1] - K[0])
    measures['lambda1_2'] = lambda1_2
    measures['lambda2_1'] = lambda2_1

    measures['deltab'] = (beta[1] - beta[0])/(epsilon*neutralbeta)
    measures['deltanu'] = (sgamma[1]- sgamma[0])/(epsilon*neutralgamma)


    weight = np.array([1/detP[0]*(-D[0]*(I[1]-I[0]) + 2*T[0]*(T[1]- T[0])) , 1/detP[1]*(-D[1]*(I[0]-I[1]) + 2*T[1]*(T[0]- T[1])) ])
    measures['weight'] = weight
    z0 = np.array([(v0[2] + v0[6] + 0.5*v0[8] + 0.5*v0[10])/(v0[2] + v0[4] + v0[6] + v0[8] + v0[10] + v0[12]), (v0[3] + v0[5] + 0.5*v0[9] + 0.5*v0[11])/(v0[3] + v0[5] + v0[7] + v0[9] + v0[11] + v0[13])])
    measures['replicator_solution'] = integrate.solve_ivp(replicator, tau, [z1[0], z1[1], 1 - z1[0], 1 - z1[1]], args = (Theta, lambda1_2, lambda2_1, weight), dense_output=True, method = 'BDF', rtol = 1e-13).y


    

    return measures



sol = analysis(system, v0, r, neutralbeta, beta, neutralgamma, sgamma, cgamma, neutralk, K, p, q, epsilon, epsilon)
def plot(sol):
    """Wrapper function to plot the solutions of the system, both in quantities and frequencies (Solutions of system and replicator, respectively)."""
    solution = sol['solution']

    labels = ['S in 1', 'S in 2', 'I1 in 1', 'I1 in 2','I2 in 1','I2 in 2', 'I11 in 1', 'I11 in 2', 'I12 in 1', 'I12 in 2', 'I21 in 1', 'I21 in 2', 'I22 in 1', 'I22 in 2']
    

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    #fig.subplots_adjust(wspace = 0.5)

    for i in range(7):
        ax[0, 0].plot(t, solution.y[2*i][:len(t)], label = labels[2*i])
        ax[0, 1].plot(t, solution.y[2*i + 1][:len(t)], label = labels[2*i+1])


    ax[1, 0].stackplot(tau[:len(sol['replicator_solution'][0])], [sol['replicator_solution'][0], sol['replicator_solution'][2]], labels = ['Strain 1', 'Strain 2'])
    ax[1, 1].stackplot(tau[:len(sol['replicator_solution'][0])], [sol['replicator_solution'][1], sol['replicator_solution'][3]], labels = ['Strain 1', 'Strain 2'])
    #Labeling everything
    for i in range(2):
        ax[0, i].set(xlabel="t")
        ax[1, i].set(xlabel=r"$\tau$")
        for j in range(2):
            ax[i, j].legend()
    ax[0, 0].set_title('Patch 1 Dynamics', fontsize = 16)
    ax[0, 1].set_title('Patch 2 Dynamics', fontsize = 16)

    return ax




plot(sol)
plt.show()

ds = np.linspace(0, 5*epsilon, 30)
sols = []
zs1 = []
zs2 = []
p1lambda1_2s = []
p1lambda2_1s = []
for d in ds:
    s = analysis(system, v0, r, neutralbeta, beta, neutralgamma, sgamma, cgamma, neutralk, K, p, q, d, epsilon)
    sols.append(s)
    zs1.append(s['replicator_solution'][0][-1])  #Now looking at solution from the replicator
    zs2.append(s['replicator_solution'][1][-1])
    p1lambda1_2s.append(s['lambda1_2'])
    p1lambda2_1s.append(s['lambda2_1'])

Animate(t, tau, ds, sols)
# test = [analysis(system, v0, r, beta, sgamma, cgamma, K, p, q, d, 0.1)['lambda1_2'], analysis(system, v0, r, beta, sgamma, cgamma, K, p, q, d, 0.1)['lambda2_1']]
# print(test[0] - test[1])

animate(ds, zs1, zs2, ['d', 'z1'], ['z1 of first patch', 'z1 of second patch'])

plt.show()


plt.stackplot(ds, zs1, label = 'z1 of first patch')
#plt.plot(ds, zs2, label = 'z1 of second patch')
plt.title('Variation of z as function of diffusion')
plt.xlabel('d')
plt.ylabel('z1')

plt.legend()
plt.show()

