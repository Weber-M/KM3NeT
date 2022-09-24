import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import Cross_section as xsec
import Flux
from super_duper_fiesta.fiesta.fiesta import nuFlux as nF

def flux(E_nu, nu, Nsteps=None, warning=False):
    '''
    Here the flux will be calculated using the power_law definition.
    The output is a list with 
        1) endindex 
        2) energy[endindex]
        3) flux obtained by the power_law as an array
    '''
    popt_list = Flux.power_law_fit(E_nu, nu, Nsteps, warning)
    startindex = 0
    F = []
    for i in range(len(popt_list)):
        if i == len(popt_list)-1 and popt_list[i][0] <= len(E_nu): # instead of != use <= and no -1
            endindex = len(E_nu)
            F.append([endindex, E_nu[endindex-1], 
                      Flux.power_law(E_nu[startindex:endindex], popt_list[i][2], popt_list[i][3])]) 
        else:
            endindex = popt_list[i][0]
            F.append([endindex, E_nu[endindex], Flux.power_law(E_nu[startindex:endindex], popt_list[i][2], 
                                                               popt_list[i][3])])
        startindex = endindex
    return F # [endindex, GeV, flux]

def dNmu_dE(element, E_nu, nu, t, Nsteps=None, warning=False):
    '''
    This definition calculates dN/dE, the change of the muonic number by energy.
    The output is the enegry and the corresponding dN/dE.
    '''
    # parameters
    V_D = 1e9 # m^3
    rho_H2O = 1e3 # kg/m^3
    M_D = V_D * rho_H2O
    N_A = 6.022*1e23 # mol^-1
    unit_change = 4 * np.pi * 1e-39 # from sr to full solid angle and fb into cm^{-2}
    
    # code
    if element == "oxygen":
        mu_H2O = 0.018 # kg/mol
        N_H2O = M_D / mu_H2O * N_A
        N_element = N_H2O
    else:
        print(f"Number of element {element} has to be added.")
        return
    
    sigma_element = xsec.QMintegration_fit(element, E_nu, Nsteps, warning)[:,1]
    F = flux(E_nu, nu, Nsteps, warning)
    startindex = 0
    dN_dE_values = np.empty((len(E_nu), 2))
    dN_dE_values[:,0] = E_nu
    for i in range(len(F)):
        endindex = F[i][0]
        dN_dE = F[i][2] * t * N_element * sigma_element[startindex:endindex] * unit_change
        dN_dE_values[startindex:endindex,1] = dN_dE
        startindex = endindex
    return dN_dE_values # [GeV, GeV^{-1}]


def number_mu(element, E_nu, nu, t, a, b, Nsteps=None, warning=False):
    '''
    This definition calculates the number of muonic events by integrating over the energy using the lower sum approximation.
    Output is the integrated dNmu_dE definition.
    '''
    dN_dE = dNmu_dE(element, E_nu, nu, t, Nsteps, warning)
    a_index = list(E_nu).index(a)
    b_index = list(E_nu).index(b)
    delta_x = dN_dE[a_index+1:b_index+1,0] - dN_dE[a_index:b_index,0]
    P = np.multiply(dN_dE[a_index:b_index,1], delta_x)
    U = np.sum(P)
    return U

def N_mu(element, E_nu, nu, t, Nsteps=None, bins=None, warning=False):
    '''
    This definition is just an extension of the number_mu definition.
    The number of steps for the calculation of the flux can be changed. Moreover, the hole energy spectrum can be divided
    into bins. For each of those the number_mu will be calculated.
    The output is 
        1) the index number of the at last used energy
        2) the at last used energy
        3) number of events for the chosen energy range
    '''
    warning_call = False
    if bins == None:
        popt_list = Flux.power_law_fit(E_nu, nu, Nsteps)
        V = [popt_list[k][1] for k in range(len(popt_list))]
        A = np.append(E_nu[0], V)
        B = np.append(V, E_nu[-1])
        endindex_energy = [popt_list[k][0] for k in range(len(popt_list))]
        endindex_energy.append(len(E_nu))
    else:
        N = int(len(E_nu) / bins)
        A = [E_nu[k*N] for k in range(bins)]
        Bb = [E_nu[(k+1)*N] for k in range(bins-1)]
        B = np.append(Bb, E_nu[-1])
        endindex_energy = [(k+1)*N for k in range(bins-1)]
        endindex_energy.append(len(E_nu))
    I_values = np.empty((len(A), 3))
    I_values[:,0] = endindex_energy
    I_values[:,1] = B
    for i in range(len(A)):
        if warning == True and i == len(A)-1:
            warning_call == True
        I = number_mu(element, E_nu, nu, t, A[i], B[i], Nsteps, warning=warning_call)
        I_values[i,2] = I
    return I_values #[index number, GeV, number of events]

def N_plothist(element, E_nu, nu, t, Nsteps=None, bins=None, warning=False):
    '''
    This definition creates a bar diagram using the values obtained from the N_mu definition.
    '''
    N = N_mu(element, E_nu, nu, t, Nsteps, bins, warning)
    N_list = np.empty((len(N), 3))
    startindex = 0
    for i in range(len(N)):
        endindex = int(N[i,0])-1
        N_list[i,0] = (E_nu[startindex] + E_nu[endindex]) / 2
        N_list[i,1] = N[i,2]
        N_list[i,2] = E_nu[endindex] - E_nu[startindex]
        startindex = endindex
    
    fig, ax = plt.subplots(figsize=(10,8))
    plt.bar(N_list[:,0], N_list[:,1], width=N_list[:,2], align='center', linewidth = 2, color='steelblue', 
            edgecolor = 'cyan')
    # This is the location for the annotated text
    # (N_list[i,0] - N_list[i,2]/2, N[i,2]*1.2)
    # Annotating the bar plot with the values
    for k in range(len(N)):
        plt.annotate(round(N[k,2],2), (N_list[k,0] - N_list[k,2]/2, N[k,2]*1.005))
    plt.legend(labels = ['Total number of events'], fontsize = 12)
    plt.title(r'$\nu_\mu \to \nu_\mu \mu^+ \mu^-$', fontsize = 25)
    plt.xlabel(r'$E_{\nu}$ (GeV)', fontsize = 20)
    plt.ylabel(r'$N$', fontsize = 20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.grid()

    plt.show()