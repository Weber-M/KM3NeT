import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from super_duper_fiesta.fiesta.fiesta import nuFlux as nF


def power_law(E_nu, phi, gamma):
    '''
    Power law with energy, a scaling phi and and exponential gamma.
    '''
    return phi * E_nu**(-gamma)

def power_law_fit(E_nu, nu, Nsteps=None, warning=False):
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
    
    if Nsteps != None:
        N = int(len(E_nu) / Nsteps)

    while start <= len(E_nu)-1:
        try:
            popt, pcov = curve_fit(power_law, E_nu[start:], nu[start:])
            if Nsteps != None:
                start += N
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
                start_byhand = len(E_nu) - 303
                if warning == True:
                    print(f"Optimal parameters not found for E > {E_nu[start]}GeV (index: >{start}) and, therefore, went back to E = {E_nu[start_byhand]}GeV (index: {start_byhand}).")
                popt, pcov = curve_fit(power_law, E_nu[start_byhand:], nu[start_byhand:])
                popt_list.append([start_byhand, E_nu[start_byhand], *popt])
                break
            else:
                if warning == True:
                    print(f"Optimal parameters not found for E > {E_nu[start]}GeV (index: >{start}).")
                break
    if warning == True:
        print(f"Fit divided into {len(popt_list)} segments.")
    return popt_list

def plot_power_law_fit_and_data(E_nu, nu, labelfordata, Nsteps=None, warning=False, E_nu__extrapolation=None):
    '''
    Uses the power_law_fit definition and makes a nice plot with data and fit.
    If E_nu__extrapolation has a value then the missing part will be added using the fit parameters colse to the energy.
    '''
    fig, ax = plt.subplots(figsize=(8,6))
    
    popt_list = power_law_fit(E_nu, nu, Nsteps, warning)
    
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
    plt.ylabel(r'F$_{\nu}$ ($GeV ^{1} \cdot cm ^{-2}\cdot s ^{-1}\cdot sr ^{-1}$)', fontsize = 20)
    plt.xlabel(r'$E$ (GeV)', fontsize = 20)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend(fontsize = 12)   
    plt.show()