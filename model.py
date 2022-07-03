import numpy as np
from scipy import integrate
import scipy.integrate  as  ode
import matplotlib.pyplot  as  plt
from animator import animate
from animation import Animate



def replicator(z, t, Theta, lambda1_2, lambda2_1, w, d):
    """Function for the replicator equation using the summarized parameters."""
    #lambda1_2 = theta1(b2 - b1) + theta2(-nu2 + nu1) + theta3*(-u21 - u12 + u11) + theta4(omega2_21 - omega1_12) + theta5*(mu(alpha12 - alpha21) + alpha12 - alpha11)
    #lambda2_1 = theta1(b2 - b1) + theta2(-nu2 + nu1) + theta3*(-u21 - u12 + u11) + theta4(omega2_21 - omega1_12) + theta5*(mu(alpha12 - alpha21) + alpha12 - alpha11)
    
    #Lambdas = [np.array([[0, lambda1_2[0]],[lambda2_1[0], 0]]), np.array([[0, lambda1_2[1]],[lambda2_1[1], 0]])]


    eqlist = []
    #Order of equations is: z11, z12. So i is refering to the strain and j is refering to the patch.
    for j in range(2):
        eqlist.append(Theta[j]*z[j]*((lambda1_2[j]*(1 - z[j])) - (lambda1_2[j]+lambda2_1[j])*(z[j]*(1-z[j]))) + (d > 0)*(w[j]-1)*(z[(j+1)%2] - z[j]))

    return eqlist

def neutralsystem(v, t, r, beta, gamma, K, p = [1, 0.5, 0.5, 1], q = [1, 0.5, 0.5, 1]):
    """Function to return the system for the neutral model."""
    n = int(len(v)/7)  #There's always 7 infection classes: S, I1, I2, I11, I12, I21, I22
    S = v[0]
    I1 = v[1]
    I2 = v[2]
    I11 = v[3]
    I12 = v[4]
    I21 = v[5]
    I22 = v[6]
    J1 = I1 + q[0]*I11 + p[0]*I11 + q[1]*I12 + p[1]*I21
    J2 = I2 + q[2]*I21 + p[2]*I12 + q[3]*I22 + p[3]*I22
    eqS = r*(1 - S) + gamma*I1 + gamma*I2 + gamma*I11 + gamma*I12 + gamma*I21 + gamma*I22 - beta*S*J1 - beta*S*J2 
    eqI1 = beta*J1*S - (r + gamma)*I1 - beta*K*I1*J1 - beta*K*I1*J2 
    eqI2 = beta*J2*S - (r + gamma)*I2 - beta*K*I2*J1 - beta*K*I2*J2 
    eqI11 = beta*K*I1*J1 - (r + gamma)*I11 
    eqI12 = beta*K*I1*J2 - (r + gamma)*I12 
    eqI21 = beta*K*I2*J1 - (r + gamma)*I21 
    eqI22 = beta*K*I2*J2 - (r + gamma)*I22 


    return [eqS, eqI1, eqI2, eqI11, eqI12, eqI21, eqI22]




def system(v, tspan, r, beta, sgamma, cgamma, K, p, q, M, d = 0):
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
    J1 = I1 + 1*I11  + p*I12 + q*I21       #I1 + q[0]*I11 + p[0]*I11 + q[1]*I12 + p[1]*I21    
    J2 = I2 + 1*I22 + (1-p)*I12 +(1-q)*I21 #I2 + q[2]*I21 + p[2]*I12 + q[3]*I22 + p[3]*I22
    eqS = r*(1 - S) + sgamma[0]*I1 + sgamma[1]*I2 + cgamma[0]*I11 + cgamma[1]*I12 + cgamma[2]*I21 + cgamma[3]*I22 - beta[0]*S*J1 - beta[1]*S*J2 + d*M@S 
    eqI1 = beta[0]*J1*S - (r + sgamma[0])*I1 - beta[0]*K[0]*I1*J1 - beta[1]*K[1]*I1*J2 + d*M@I1
    eqI2 = beta[1]*J2*S - (r + sgamma[1])*I2 - beta[0]*K[2]*I2*J1 - beta[1]*K[3]*I2*J2 + d*M@I2
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


def solve(system, t, v0, r, beta, sgamma, cgamma, K, p, q, M, d = 0):
    return integrate.odeint(system, v0, t, args = (r, beta, sgamma, cgamma, K, p, q, M,  d))



def analysis(system, tspan, v0, r, neutralbeta, b, neutralgamma, sgamma, cgamma, neutralk,  K, p, q, M, d, epsilon):
    """Function to perform the analysis of the model. Inputs are initial conditions and parameters of the model and an option to plot."""
    m = r + neutralgamma
    measures = dict()
    R0 = neutralbeta/m
    measures['R_0'] = R0
    measures['epsilon'] = epsilon
    solution = solve(system, tspan, v0, r, b, sgamma, cgamma, K, p, q, M, d)
    measures['solution'] = solution

    I1 = np.array([solution.T[2], solution.T[3]])
    I2 = np.array([solution.T[4], solution.T[5]])
    T = np.array([solution.T[2] + solution.T[4] + solution.T[6]  + solution.T[8] + solution.T[10] + solution.T[12], solution.T[3] + solution.T[5] + solution.T[7]  + solution.T[9] + solution.T[11] + solution.T[13]])
    I = I1 + I2
    D = T - I
    z1 = np.array([solution.T[2] + solution.T[6] + 0.5*(solution.T[8] + solution.T[10]), solution.T[3] + solution.T[7] + 0.5*(solution.T[9] + solution.T[11])])/T
    z2 = np.array([solution.T[4] + solution.T[12] + 0.5*(solution.T[8] + solution.T[10]), solution.T[5] + solution.T[13] + 0.5*(solution.T[9] + solution.T[11])])/T
    measures['I'] = I
    measures['S'] = [solution.T[0], solution.T[1]]
    measures['T'] = T
    measures['D'] = D
    measures['z1'] = z1
    measures['z2'] = z2
    

    #Theoritical equilibria
    TSstar = 1/R0
    measures['TSstar'] = TSstar
    TTstar = 1 - TSstar
    measures['TTstar'] = TTstar
    TIstar = TTstar/(1 + neutralk*(R0 - 1))
    TDstar = TTstar - TIstar
    measures['TIstar'] = TIstar
    measures['TDstar'] = TDstar

    Sstar = np.array([solution.T[0][-1], solution.T[1][-1]])
    I1star =  np.array([solution.T[2][-1], solution.T[3][-1]])
    I2star =  np.array([solution.T[4][-1], solution.T[5][-1]])
    I11star = np.array([solution.T[6][-1], solution.T[7][-1]])
    I12star = np.array([solution.T[8][-1], solution.T[9][-1]])
    I21star = np.array([solution.T[10][-1], solution.T[11][-1]])
    I22star = np.array([solution.T[12][-1], solution.T[13][-1]])
    measures['I1*'] = I1star
    measures['I2*'] = I2star
    measures['I11*'] = I11star
    measures['I12*'] = I12star
    measures['I21*'] = I21star
    measures['I22*'] = I22star
    Tstar = I1star + I2star + I11star + I12star + I21star + I22star
    Istar = I1star + I2star
    Dstar = Tstar - Istar
    measures['Sstar'] = Sstar
    measures['Tstar'] = Tstar
    measures['Istar'] = Istar
    measures['Dstar'] = Dstar
    detP = np.array([np.linalg.det([[2*TTstar[0], TIstar[0]], [TDstar[0], TTstar[0]]]), np.linalg.det([[2*TTstar[1], TIstar[1]], [TDstar[1], TTstar[1]]])])
    # z1 = np.array([(I1star + I11star + 0.5*I21star + +0.5*I12star)[0]/Tstar[0], (I1star + I11star + 0.5*I21star + 0.5*I12star)[1]/Tstar[1]])
    # z2 = np.array([(I2star + I22star + 0.5*I21star + +0.5*I12star)[0]/Tstar[0], (I2star + I22star + 0.5*I21star + 0.5*I12star)[1]/Tstar[1]])
    # measures['z1'] = z1
    # measures['z2'] = z2

    #Implement invasion fitness lambda_i_j for each patch
    #These are still vectors (one entry for each patch)     
    Theta1 = (2*neutralbeta*TSstar*Tstar**2)/detP
    Theta2 = neutralgamma*TIstar*(TIstar + TTstar)/detP
    Theta3 = neutralgamma*TTstar*TDstar/detP
    Theta4 = 2*m*TTstar*TDstar/detP
    Theta5 = neutralbeta*TTstar*TIstar*TDstar/detP
    Theta = Theta1 + Theta2 + Theta3 + Theta4 + Theta5
    theta1 = Theta1/Theta
    theta2 = Theta2/Theta
    theta3 = Theta3/Theta
    theta4 = Theta4/Theta
    theta5 = Theta5/Theta
    mu = Istar/Dstar
    measures['mu'] = mu

    #This needs to be done with b_i
    lambda1_2 = theta1*(b[0] -  b[1]) + theta2*(sgamma[0] - sgamma[1]) + theta3*(-cgamma[1] - cgamma[2] + 2*cgamma[3]) + theta4*((q[0] - p[1])/epsilon) + theta5*(mu*(K[2] - K[1]) + K[2] - K[3])
    lambda2_1 = theta1*(b[1] - b[0]) + theta2*(sgamma[1] - sgamma[0]) + theta3*(-cgamma[2] - cgamma[1] + 2*cgamma[0]) + theta4*((q[0] - p[1])/epsilon) + theta5*(mu*(K[1] - K[2]) + K[1] - K[0])
    measures['lambda1_2'] = lambda1_2
    measures['lambda2_1'] = lambda2_1

    measures['deltab'] = (b[1] - b[0])/(epsilon*neutralbeta)
    measures['deltanu'] = (sgamma[1]- sgamma[0])/(epsilon*neutralgamma)


    w = np.array([1/detP[0]*(-TDstar[0]*(TIstar[1]-TIstar[0]) + 2*TTstar[0]*(TTstar[1]- TTstar[0])) , 1/detP[1]*(-TDstar[1]*(TIstar[0]-TIstar[1]) + 2*TTstar[1]*(TTstar[0]- TTstar[1])) ])
    measures['w'] = w
    z0 = np.array([(v0[2] + v0[6] + 0.5*v0[8] + 0.5*v0[10])/(v0[2] + v0[4] + v0[6] + v0[8] + v0[10] + v0[12]), (v0[3] + v0[5] + 0.5*v0[9] + 0.5*v0[11])/(v0[3] + v0[5] + v0[7] + v0[9] + v0[11] + v0[13])])


    repli, info = integrate.odeint(replicator, [z1[0][0], z1[1][0]], epsilon*tspan, args = (Theta, lambda1_2, lambda2_1, w, d), full_output=True)
    measures['replicator_solution'] = repli

    
    measures['error'] = np.linalg.norm(I[0]*repli.T[0] - I1[0]) #+ np.linalg.norm(I[1]*repli.T[0] - I1[1]))/2
    

    return measures



def plot(sol, tspan):
    """Wrapper function to plot the solutions of the system, both in quantities and frequencies (Solutions of system and replicator, respectively)."""
    solution = sol['solution']

    labels = ['S', 'S', 'I1', 'I1','I2','I2', 'I11', 'I11', 'I12', 'I12', 'I21', 'I21', 'I22', 'I22']
    


    
    fig, ax = plt.subplots(3, 2, figsize = (10, 10))

    ax[1, 0].set_ylim([0, 1])
    ax[1, 1].set_ylim([0, 1])

    for i in range(7):
        ax[0, 0].plot(tspan, solution.T[2*i], label = labels[2*i])
        ax[0, 1].plot(tspan, solution.T[2*i + 1], label = labels[2*i+1])
    ax[1, 0].plot(tspan, sol['z1'][0], label = 'Strain 1')
    ax[1, 0].plot(tspan, sol['replicator_solution'].T[0], '--', label = 'replicator z1')
    #ax[1, 0].plot(tspan, sol['z2'][0], label = 'Strain 2')
    #ax[1, 0].plot(tspan, np.ones(tspan) - sol['replicator_solution'].T[1], '--', label = 'replicator z2')
    #ax[1, 0].plot(tspan, sol['replicator_solution'][:, 0], label = 'replicator strain 1')
    #ax[1, 0].plot(tspan, sol['replicator_solution'][:, 1], label = 'replicator strain 2')

    ax[1, 1].plot(tspan, sol['z1'][1], label = 'Strain 1')
    ax[1, 1].plot(tspan, sol['replicator_solution'].T[1], '--', label = 'replicator z1')
    #ax[1, 1].plot(tspan, sol['z2'][1], label = 'Strain 2')
    #ax[1, 1].plot(tspan, np.ones(tspan) - sol['replicator_solution'].T[1], '--', label = 'replicator z2')
    #ax[1, 1].plot(tspan, sol['replicator_solution'][:, 2], label = 'replicator solution 1')
    #ax[1, 1].plot(tspan, sol['replicator_solution'][:, 3], label = 'replicator solution 2')
    
    

    ax[2, 0].plot(tspan, sol['S'][0], label = 'S', color = 'blue')
    ax[2, 0].plot(tspan, [sol['TSstar'][0] for t in tspan],'--', label = '$S^*$', color = 'blue')
    ax[2, 0].plot(tspan, sol['I'][0], label = 'I', color = 'red')
    ax[2, 0].plot(tspan, [sol['TIstar'][0] for t in tspan],'--', label = '$I^*$', color = 'red')
    ax[2, 0].plot(tspan, sol['D'][0], label = 'D', color = 'orange')
    ax[2, 0].plot(tspan, [sol['TDstar'][0] for t in tspan],'--', label = '$D^*$', color = 'orange')
    ax[2, 0].plot(tspan, sol['T'][0], label = 'T', color = 'purple')
    ax[2, 0].plot(tspan, [sol['TTstar'][0] for t in tspan],'--', label = '$T^*$', color = 'purple')

    ax[2, 1].plot(tspan, sol['S'][1], label = 'S', color = 'blue')
    ax[2, 1].plot(tspan, [sol['TSstar'][1] for t in tspan],'--', label = '$S^*$', color = 'blue')
    ax[2, 1].plot(tspan, sol['I'][1], label = 'I', color = 'red')
    ax[2, 1].plot(tspan, [sol['TIstar'][1] for t in tspan],'--', label = '$I^*$', color = 'red')
    ax[2, 1].plot(tspan, sol['D'][1], label = 'D',  color = 'orange')
    ax[2, 1].plot(tspan, [sol['TDstar'][1] for t in tspan],'--', label = '$D^*$',  color = 'orange')
    ax[2, 1].plot(tspan, sol['T'][1], label = 'T', color = 'purple')
    ax[2, 1].plot(tspan, [sol['TTstar'][1] for t in tspan],'--', label = '$T^*$', color = 'purple')



    #Labeling everything
    ax[0, 0].text(91, 0.455, 'R0 = ' + str(round(sol['R_0'][0], 3)), style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, ha='center', va='center')
    ax[0, 1].text(86, 0.5, 'R0 = ' + str(round(sol['R_0'][1], 3)), style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, ha='center', va='center')
    ax[1, 0].text(len(tspan)/2, 0.4, '$\lambda_1^2$ = ' + str(round(sol['lambda1_2'][0], 3)) + '\n $\lambda_2^1$ =' + str(round(sol['lambda2_1'][0], 3)), style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, ha='center', va='center')
    ax[1, 1].text(len(tspan)/2, 0.4, '$\lambda_1^2$ = ' + str(round(sol['lambda1_2'][1], 3)) + '\n $\lambda_2^1$ =' + str(round(sol['lambda2_1'][1], 3)), style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, ha='center', va='center')
    for i in range(2):
        ax[0, i].set(xlabel="t")
        ax[1, i].set(xlabel=r"$\tau$")
        ax[i, 1].legend(loc = 'center right', bbox_to_anchor=(1.25, 0.5))
        ax[2, 1].legend(loc = 'center right', bbox_to_anchor=(1.25, 0.5))

    ax[0, 0].set_title('Patch 1 Dynamics', fontsize = 16)
    ax[0, 1].set_title('Patch 2 Dynamics', fontsize = 16)

    fig.suptitle('Two patch dynamics with mean R0 =' + str(round(sum(sol['R_0'])/2, 3)) + ' and $\epsilon$ =' + str(sol['epsilon']))


    return ax



