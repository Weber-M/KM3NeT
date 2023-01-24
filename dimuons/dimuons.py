import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import Cross_section as xsec
import Flux
from super_duper_fiesta.fiesta.fiesta import nuFlux as nF


def power_law(E_nu, a, b):
    return np.exp(a + b * np.log(E_nu))

def power_law_fit(E_nu, nu, Nsteps = None):
    '''
    This definition finds for all energies the fit parameters and puts them into a list.
    There a two options:
        - it checks how much the deviation is and if it is to much the fit paramaters will be calculated again
        - if Nsteps has a value then the fit will be divided into Nsteps 
    Output is a list with 
        1) ending index (tells to which energy the deviation was small enough)
        2) corresponding energy to the ending index
        3) parameters for the fitted function
    '''
    start = 0
    popt_list = []
    end = -1
    
    if Nsteps != None:
        N = int(len(E_nu) / Nsteps)
        end = N

    while start <= len(E_nu)-1:
        try:
            popt, pcov = curve_fit(power_law, E_nu[start:end], nu[start:end])
            if Nsteps != None:
                start += N
                end += N
                if start >= len(E_nu):
                    start = len(E_nu)-1
                popt_list.append([start, E_nu[start], *popt])
                
            else:
                for i in range(len(E_nu[start:])):
                    a = power_law(E_nu[start:], *popt)[i] / nu[start:][i] 
                    if a <= 0.99868 or a >= 1.1606:
                        start += i
                        popt_list.append([start, E_nu[start], *popt])
                        break
        except:
            if Nsteps != None and len(popt_list) != Nsteps:
                start_byhand = len(E_nu) - int(len(E_nu)/3.29)
                print(f"Optimal parameters not found for E > {E_nu[start]}GeV (index: >{start}) and, therefore, went back to E = {E_nu[start_byhand]}GeV (index: {start_byhand}).")
                popt, pcov = curve_fit(power_law, E_nu[start_byhand:], nu[start_byhand:])
                popt_list.append([start_byhand, E_nu[start_byhand], *popt])
                break
            else:
                print(f"Optimal parameters not found for E > {E_nu[start]}GeV (index: >{start}).")
                break
    print(f"Fit divided into {len(popt_list)} segments.")
    return popt_list

def plot_power_law_fit_and_data(E_nu, nu, labelfordata, Nsteps = None, E_nu__extrapolation = None):
    '''
    Uses the power_law_fit definition and makes a nice plot with data and fit.
    If E_nu__extrapolation has a value then the missing part will be added using the fit parameters colse to the energy.
    '''
    fig, ax = plt.subplots(figsize=(8,6))
    
    popt_list = power_law_fit(E_nu, nu, Nsteps)
    
    ax.plot(E_nu, nu, label = f"{labelfordata}")

    startindex = 0
    for i in range(len(popt_list)):
        if i == len(popt_list)-1 and popt_list[i][0]-1 != len(E_nu):
            endindex = len(E_nu)
            ax.plot(E_nu[startindex:endindex], power_law(E_nu[startindex:endindex], popt_list[i][2], popt_list[i][3]), 'r--',
                label = "fit")
        else:
            endindex = popt_list[i][0]-1
            ax.plot(E_nu[startindex:endindex], power_law(E_nu[startindex:endindex], popt_list[i][2], popt_list[i][3]), 'r--')
        startindex = endindex+1
    
    if E_nu__extrapolation != None:
        if E_nu__extrapolation >= E_nu[0] and E_nu__extrapolation <=  E_nu[-1]:
            L = np.array([E_nu__extrapolation for i in range(len(E_nu))])
            diff = np.abs(L - E_nu)
            index = np.array([list(diff).index(min(diff)) for i in range(len(popt_list))])
            popt_list_index = np.array([popt_list[i][0] for i in range(len(popt_list))])
            diff_index_array = np.abs(index - popt_list_index)
            diff_index = list(diff_index_array).index(min(diff_index_array))
            
            ax.plot(E_nu__extrapolation, power_law(E_nu__extrapolation, popt_list[diff_index][2], popt_list[diff_index][3]),
                    'o', color = 'green', label = f"{E_nu__extrapolation}GeV")
        else:
            if E_nu__extrapolation < E_nu[0]:
                index = popt_list.index(popt_list[0])
            if E_nu__extrapolation > E_nu[-1]:
                index = popt_list.index(popt_list[-1])
            ax.plot([E_nu__extrapolation, E_nu[index]], power_law([E_nu__extrapolation, E_nu[index]], popt_list[index][2],
                                                                  popt_list[index][3]), 'g--')

    plt.title(r'$\nu_{\mu}$', fontsize = 25)
    plt.ylabel(r'F$_{\nu}$ ($GeV ^{-1} \cdot cm ^{-2}\cdot s ^{-1}\cdot sr ^{-1}$)', fontsize = 20)
    plt.xlabel(r'$E$ (GeV)', fontsize = 20)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend(fontsize = 12)   
    plt.show()
    
def xsec_Ice(E_nu, Nsteps = None):
    P = power_law_fit(Ice[:,0], Ice[:,1], Nsteps)
    xsec_list = []
    for i in range(len(P)+1):
        for j in E_nu:
            if i == 0 and j <= P[i][1]:
                L = power_law(j, P[i][2], P[i][3])
                xsec_list.append(L)
            elif i == len(P) and j >= P[i-1][1]:
                L = power_law(j, P[i-1][2], P[i-1][3])
                xsec_list.append(L)
            elif (i != 0 and i != len(P)) and j <= P[i][1] and j >= P[i-1][1]:
                L = power_law(j, P[i][2], P[i][3])
                xsec_list.append(L)
    return xsec_list

def dNmu_dE_IceCube(element, E_nu, nu, t, Nsteps=None, warning=True):
    '''
    This definition calculates dN/dE, the change of the muonic number by energy.
    The output is the enegry and the corresponding dN/dE.
    '''
    # parameters
    V_D = 1e9 # m^3
    N_A = 6.022*1e23 # mol^-1
    unit_change = 4 * np.pi # from sr to full solid angle and fb into cm^{-2}
    
    # code
    if element == "oxygen":
        mu_H2O = 0.018 # kg/mol
        rho_H2O = 1e3 # kg/m^3
        rho = rho_H2O
        M_D = V_D * rho
        N_H2O = M_D / mu_H2O * N_A
        N_element = N_H2O
    elif element == "ice":
        mu_H2O = 0.018 # kg/mol
        rho_ice = 920 # kg/m^3
        rho = rho_ice
        M_D = V_D * rho
        N_ice = M_D / mu_H2O * N_A
        N_element = N_ice
    else:
        print(f"Number of element {element} has to be added.")
        return
    
    sigma_element = xsec_Ice(E_nu, Nsteps)
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

def number_mu_IceCube(element, E_nu, nu, t, a, b, Nsteps=None, warning=True):
    '''
    This definition calculates the number of muonic events by integrating over the energy using the lower sum approximation.
    Output is the integrated dNmu_dE definition.
    '''
    dN_dE = dNmu_dE_IceCube(element, E_nu, nu, t, Nsteps, warning)
    a_index = list(E_nu).index(a)
    b_index = list(E_nu).index(b)
    delta_x = dN_dE[a_index+1:b_index+1,0] - dN_dE[a_index:b_index,0]
    P = np.multiply(dN_dE[a_index:b_index,1], delta_x)
    U = np.sum(P)
    return U

def N_mu_IceCube(element, E_nu, nu, t, Nsteps=None, bins=None, warning=True):
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
        I = number_mu_IceCube(element, E_nu, nu, t, A[i], B[i], Nsteps, warning=warning_call)
        I_values[i,2] = I
    return I_values #[index number, GeV, number of events]

def N_plothist_IceCube(element, E_nu, nu, t, Nsteps=None, bins=None, warning=True):
    '''
    This definition creates a bar diagram using the values obtained from the N_mu definition.
    '''
    N = N_mu_IceCube(element, E_nu, nu, t, Nsteps, bins, warning)
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
#     plt.yscale('log')
    plt.xscale('log')
    # plt.ylim()
    # plt.xlim()
    plt.grid()

    plt.show()