import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def sigma_mu(A, Z, E_nu):
    '''
    This definition calculates the cross section for the neutrino trident event with muons as an outcome.
    The input is simply the mass number ([A] = 0) and proton number ([Z] = 0) of the traget nulceus and the neutrion beam 
    energy ([E_nu] = GeV). 
    The output is obviously the cross section value.
    '''
    onedivfm_to_GeV = 197*10**(-3) 
    onedivGeV_to_fm = onedivfm_to_GeV
    alpha = 1/137
    G = 1.1663787*10**(-5) #[GeV^(-2)]
    beta = np.sqrt(5)/(1.2*A**(1/3)) * onedivfm_to_GeV #[fm^(-1)]->[GeV]
    C = 0.577
    m_mu = 105.6583755*10**(-3) #[GeV]
    
    sigma = (4 * Z**2 * alpha**2 * G**2 / (9 * np.pi**3) * E_nu * beta * np.sqrt(np.pi) / 2 * 
            (4 / 3 * np.log(2 * E_nu * beta / m_mu**2) - (16 / 3 + 2 / 3 * C + 4 / 3 * np.log(2) + 1 / 15 * 
              (beta / m_mu)**2)))
    return sigma * (onedivGeV_to_fm)**2 * 10**(13) #[GeV^(-2)]->[fb]

def sigma_mu_list(A, Z, E_nu):
    '''
    This definition gives all sigmas for different E_nu's as an output in a list or a number if the input is just a number.
    '''
    if type(E_nu)!=list and type(E_nu)!=np.ndarray:
        sigma_values = sigma_mu(A, Z, E_nu)
    else:
        sigma_values = []
        for E in E_nu:
            sigma_values.append(sigma_mu(A, Z, E))
    return sigma_values #[fb]

def ratio_sigma_mu(A_1, Z_1, A_2, Z_2, E_nu, default_list=False):
    '''
    This definition calculates the ratios between two curves obtained by the sigma_mu definition.
    The input is simply the mass number ([A_1/2] = 0) and proton number ([Z_1/2] = 0) of the traget nulcei and the neutrion 
    beam energy ([E_nu] = GeV) where the differences want to be observed.
    '''
    curve1 = sigma_mu_list(A_1, Z_1, E_nu)
    curve2 = sigma_mu_list(A_2, Z_2, E_nu)
    if type(curve1)!=list and type(curve1)!=np.ndarray:
        ratio_value = curve1 / curve2
    else:
        ratio_value = []
        for i in range(len(curve1)):
            ratio = curve1[i] / curve2[i]
            ratio_value.append(ratio)      
    if default_list==True:
        return ratio_value
    else:
        return np.mean(ratio_value)
    
def ratio_sigma_mu_data(A, Z, data_list, E_nu, default_dataovercurve=True, default_list=False):
    '''
    This definition calculates the ratios between data and curve obtained by the sigma_mu definition.
    The input is simply the mass number ([A] = 0) and proton number ([Z] = 0) of the traget nulceus, the data (first colum has
    to be the E_nu and the second one the sigma_mu) referring to the nucleus and the neutrion beam energy ([E_nu] = GeV) where 
    the differences want to be observed.
    Important: data_list has to be a list!
    '''
    E_all = data_list[:,0]
    E = []
    for V in E_all:
        if V >= E_nu:
            E.append(V)
    curve = sigma_mu_list(A, Z, E)
    if type(curve)!=list and type(curve)!=np.ndarray:
        ratio_value = data_list[1] / curve
    else:
        ratio_value = []
        for i in range(len(curve)):
            ratio = data_list[i,1] / curve[i]
            if default_dataovercurve==False:
                ratio_value.append(1/ratio)
            else:
                ratio_value.append(ratio)
    if default_list==True:
        return ratio_value
    else:
        return np.mean(ratio_value)

def power_law_xsec(E_nu, a, b):
    '''
    Power law for the cross section with the energy E_nu as an input.
    '''
    return np.exp(a + b * np.log(E_nu))

def power_law_fit_xsec(E_nu, sigma, Nsteps=None, warning=False):
    '''
    This definition finds for all energies the fit parameters and puts them into a list.
    There a two options:
        - it checks how much the deviation is and if it is to much the fit paramaters will be calculated again
        - if Nsteps has a value then the fit will be divided into Nsteps 
    Output is a list with 
        1) ending index (tells till which energy the deviation was small enough)
        2) corresponding energy to the ending index
        3) parameters for the fitted function
    '''
    start = 0
    c = 0
    popt_list = []
    
    if Nsteps != None:
        N = int(len(E_nu) / Nsteps)
    else:
        N = int(len(E_nu) / 10)
        
    end = N

    while start <= len(E_nu)-1:
        try:
            popt, pcov = curve_fit(power_law_xsec, E_nu[start:end], sigma[start:end])
            start += N
            end += N
            if start >= len(E_nu):
                start = len(E_nu)-1
            popt_list.append([start, E_nu[start], *popt])

        except:
            if Nsteps != None and len(popt_list) != Nsteps:
                if c == 0:
                    start_byhand = start - 1
                try:
                    popt, pcov = curve_fit(power_law_xsec, E_nu[start_byhand:], sigma[start_byhand:])
                except:
                    start_byhand -= 1
                    c = 1
                    break
                popt_list.append([len(E_nu)-1, E_nu[-1], *popt])
                if warning == True:
                    print(f"Optimal parameters not found for E > {E_nu[start]}GeV (index: >{start}) and, therefore, went back to E = {E_nu[start_byhand]}GeV (index: {start_byhand}).")
                break
            else:
                if warning == True:
                    print(f"Optimal parameters not found for E > {E_nu[start]}GeV (index: >{start}).")
                break
    if warning == True:
        print(f"Fit divided into {len(popt_list)} segments.")
    return popt_list # [index, energy (GeV), fitting values (a, b)]

def sigma_mu_QM(element, E_nu, Nsteps=None, warning=False):
    '''
    This definition calculates the fitting paramters for the given QM data and gives as an oputput all possible values which
    are desired.
    The input is simply the name of the element and the energy steps/values which are desired, 
    e.g. E_nu = np.linspace(0, 1e3, 100000). Here, only the values n the range of 1GeV-100GeV will be used and calculated.
    Note: this approximation is only vaild for the range of 1GeV-100GeV, below and above the deviation is to high.
    The output:
        - energy (GeV)
        - corresponding corss section (fb)
    '''
    A_argon = 40 
    Z_argon = 18
    data_argon = np.loadtxt('C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Ar-table.txt')
    
    A_ferrum = 56
    Z_ferrum = 26
    data_ferrum = np.loadtxt("C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Fe-table.txt")

    A_oxygen = 18
    Z_oxygen = 8
    e = np.linspace(20, 130, 10000)
    data_oxygen = np.zeros((len(data_ferrum),2)) # approximation from ferrum
    data_oxygen[:,1] = data_ferrum[:,1]*ratio_sigma_mu(A_oxygen, Z_oxygen, A_ferrum, Z_ferrum, e)
    data_oxygen[:,0] = data_ferrum[:,0]
    
    if element == "argon":
        data = data_argon
    elif element == "ferrum":
        data = data_ferrum
    elif element == "oxygen":
        data = data_oxygen
    else:
        print(f"Element {element} is not implemented in the code.")
        return
    
    E1e2 = [] # finding all values which are in the range of 1GeV-100GeV
    for i in E_nu:
        if i >= 1 and i <= 1e2:
            E1e2.append(i)
            
    P = power_law_fit_xsec(data[:,0], data[:,1], Nsteps, warning)
    xsec_values = np.empty((len(E1e2), 2))
    xsec_list = []
    for i in range(len(P)+1):
        for j in E1e2:
            if i == 0 and j <= P[i][1]:
                L = power_law_xsec(j, P[i][2], P[i][3])
                xsec_list.append(L)
            elif i == len(P) and j > P[i-1][1]:
                L = power_law_xsec(j, P[i-1][2], P[i-1][3])
                xsec_list.append(L)
            elif (i != 0 and i != len(P)) and j <= P[i][1] and j > P[i-1][1]:
                L = power_law_xsec(j, P[i][2], P[i][3])
                xsec_list.append(L)
    xsec_values[:,0] = E1e2
    xsec_values[:,1] = xsec_list
    return xsec_values # [E1e2 (GeV), xsec (fb)]

def sigma_mu_list_shifted(element, E_nu, Nsteps=None, warning=False):
    '''
    Here, the sigma_mu_list definition will be scaled to complete the cross section obtained by QM integration.
    Output is the hole sigma_mu_list definition shifted by the factor 
    sigma_mu_QM[at the given strating energy] / sigma_mu_list[at the given starting energy].
    '''
    E_nu_modified = E_nu
    counter = 0
    for i in E_nu:
        if i <= 1e2:
            counter += 1
    if counter == 0:
        E_nu_modified = [1e2] # to get the ratio even if the interesting energy range is above 1e2GeV
        
    xsec = sigma_mu_QM(element, E_nu_modified, Nsteps)
    
    if element == "argon":
        A = 40
        Z = 18
    elif element == "ferrum":
        A = 56
        Z = 26
    elif element == "oxygen":
        A = 18
        Z = 8
    else:
        print(f"Element {element} is not implemented in the code.")
        return
    return np.multiply(sigma_mu_list(A, Z, E_nu), np.divide(xsec[-1][1], sigma_mu_list(A, Z, xsec[-1][0])))

def QMintegration_fit(element, E_nu, Nsteps=None, warning=True):
    '''
    This definition approximates the theory for the cross section obtained by the QM integration. 
    Starting with a polynomial fit the approximation goes on by using the sigma_mu_list_shiftedwithpoly definition.
    Output is a list with the energy (values[:,0]) and the corresponding cross section (values[:,1]).
    '''
    values = np.empty((len(E_nu), 2))
    
    if element != "argon" and element != "ferrum" and element != "oxygen":
        print(f"Element {element} is not implemented in the code.")
        return
    
    xsec_QM = sigma_mu_QM(element, E_nu, Nsteps, warning)[:,1]
    Egreater1 = []
    for i in E_nu:
        if i >= 1:
            Egreater1.append(i)
    values[:,0] = Egreater1
    xsec = np.append(xsec_QM, sigma_mu_list_shifted(element, Egreater1, Nsteps)[len(xsec_QM):])
    values[:,1] = xsec
    return values # [GeV, fb]
    
def sigma_mu_list_shiftedwithpoly(A, Z, E):
    '''
    Here, the sigma_mu_list definition will be scaled to complete the cross section obtained by QM integration.
    Output is the hole sigma_mu_list definition shifted by the factor 
    poly[at the given strating energy] / sigma_mu_list[at the given starting energy].
    '''
    A_argon = 40 
    Z_argon = 18
    
    A_ferrum = 56
    Z_ferrum = 26
    
    A_oxygen = 18
    Z_oxygen = 8
    
    if A == A_argon:
        data_argon = np.loadtxt('C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Ar-table.txt')
        poly = np.poly1d(np.polyfit(data_argon[:,0], data_argon[:,1], 20))
    if A == A_ferrum:
        data_ferrum = np.loadtxt("C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Fe-table.txt")
        poly = np.poly1d(np.polyfit(data_ferrum[:,0], data_ferrum[:,1], 15))
    if A == A_oxygen:
        e = np.linspace(20, 130, 10000)
        data_ferrum = np.loadtxt("C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Fe-table.txt")
        data_oxygen = np.zeros((len(data_ferrum),2)) # approximation from ferrum
        data_oxygen[:,1] = data_ferrum[:,1]*ratio_sigma_mu(A_oxygen, Z_oxygen, A_ferrum, Z_ferrum, e)
        data_oxygen[:,0] = data_ferrum[:,0]
        poly = np.poly1d(np.polyfit(data_oxygen[:,0], data_oxygen[:,1], 15))
    return np.multiply(sigma_mu_list(A, Z, E), np.divide(poly(E[0]), sigma_mu_list(A, Z, E[0])))

def QMintegration_fit_poly(element, E_nu):
    '''
    This definition approximates the theory for the cross section obtained by the QM integration. 
    Starting with a polynomial fit the approximation goes on by using the sigma_mu_list_shiftedwithpoly definition.
    Output is a list with the energy (values[:,0]) and the corresponding cross section (values[:,1]).
    '''
#     values = np.zeros((len(E_nu), 2))
    values = np.empty((len(E_nu), 2))
    N = len(E_nu)
    M = int(N / 100)
    h = np.linspace(E_nu[0], 1e2, M)
    H = np.linspace(1e2, E_nu[-1], N - M)
    
    A_argon = 40 
    Z_argon = 18
    data_argon = np.loadtxt('C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Ar-table.txt')

    A_ferrum = 56
    Z_ferrum = 26
    data_ferrum = np.loadtxt("C:/Users/xxMax/OneDrive/Desktop/UNI/Bachelor&Master/8.Semester/ResearchProject/Coding/Pictures/cross_section_mu_in_Fe-table.txt")

    A_oxygen = 18
    Z_oxygen = 8
    e = np.linspace(20, 130, 10000)
    data_oxygen = np.zeros((len(data_ferrum),2)) # approximation from ferrum
    data_oxygen[:,1] = data_ferrum[:,1]*ratio_sigma_mu(A_oxygen, Z_oxygen, A_ferrum, Z_ferrum, e)
    data_oxygen[:,0] = data_ferrum[:,0]

    
    if element == "argon":
        poly = np.poly1d(np.polyfit(data_argon[:,0], data_argon[:,1], 20))
        A = A_argon
        Z = Z_argon
    elif element == "ferrum":
        poly = np.poly1d(np.polyfit(data_ferrum[:,0], data_ferrum[:,1], 15))
        A = A_ferrum
        Z = Z_ferrum
    elif element == "oxygen":
        poly = np.poly1d(np.polyfit(data_oxygen[:,0], data_oxygen[:,1], 15))
        A = A_oxygen
        Z = Z_oxygen
    else:
        print(f"Element {element} is not implemented in the code.")
        return 
    
    energies = np.append(h, H)
    values[:,0] = energies
    xsec = np.append(poly(h), sigma_mu_list_shiftedwithpoly(A, Z, H))
    values[:,1] = xsec
    return values #[GeV, fb]