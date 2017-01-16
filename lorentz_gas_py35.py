import math
import string
#import bottleneck as bn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import time
import datetime
from scipy import interpolate
from scipy import integrate
from scipy import constants
from scipy import special
from collections import namedtuple
import cProfile

# Function for kernels
def smooth(x,window_len=15,window='bartlett'):
        if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

length = 301 #Number of data points
scale = 0.1 #Step size

# Pressure conversion dictionary
p_conversion = {'torr' : constants.torr,
                'mmHg' : constants.torr,
                'Pascal' : 1,
                'Pa' : 1,
                'atm' : constants.atmosphere,
                'bar' : constants.bar}

# Isotope shorthand conversion

isotope_string = { 'H' : '1H',
                    'D' : '2H',
                    'C' : '12C',
                    'E' : '13C',
                    'O' : '16O',
                    'P' : '17O',
                    'Q' : '18O'}

# (Incomplete) list of isotopic mass definitions from http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
# In amu

isotope_masses = { '1H': 1.00782503207,
                    '2H': 2.0141017778,
                    '12C': 12.0000000,
                    '13C': 13.0033548378,
                    '16O': 15.99491461956,
                    '17O': 16.99913170,
                    '18O': 17.9991610}

# Oxygen isotope standard dictionary
isotope_conversion = {'VSMOW': (3.8672E-4, 2.0052E-3),
                    'VPDB': (3.9511E-4, 2.08835E-3)}
                    

# Concentration conversion dictionary
conc_conversion = {'cm^-3': 1E6,
                    'm^-3': 1.}

oxygen_isotopes = ['16O', '17O', '18O']
rare_oxygen_isotopes = ['17O', '18O']
other_isotopes = ['1H', '2H', '12C', '13C']
oxygen_states = ('3P', '1D', '1S')

# Possible isotopic variants of gases
O2_combinations = ('16O16O', '17O16O', '18O16O')
CO2_combinations = ('12C16O16O', '12C17O16O', '12C18O16O')

def convert_isotope_shorthand(formula):
    """Create isotope formula for shorthand. Note: currently not used in code"""
    isotope_formula = ''
    for char in formula:
        if char in isotope_string.keys():
            new_char = isotope_string[char]
            isotope_formula += new_char
        elif char in string.digits:
            digit = int(char)
            isotope_formula += (digit-1)*new_char
    return isotope_formula

def maxwell_boltzmann(energy):
    """Returns a maxwell boltzmann distribution for a given energy space"""
    return 2*np.sqrt(energy/math.pi) * np.exp(-energy)

def boltzmann_kernel(x,x_pri,Q,R):
    """Calculates unitless kernel for probability of energy transfer from
    x_pri to x given unitless mass ratios Q and R
    """
    if x_pri > x:
        return math.erf(Q*x**0.5 + R*x_pri**0.5) + math.erf(Q*x**0.5 - R*x_pri**0.5) +\
        math.exp(x_pri - x)*(math.erf(R*x**0.5 + Q*x_pri**0.5) +\
        math.erf(R*x**0.5 - Q*x_pri**0.5))
    elif x_pri <= x:
        return math.erf(Q*x**0.5 + R*x_pri**0.5) - math.erf(Q*x**0.5 - R*x_pri**0.5) +\
        math.exp(x_pri - x)*(math.erf(R*x**0.5 + Q*x_pri**0.5) -\
        math.erf(R*x**0.5 - Q*x_pri**0.5))
        
def collision_gain_integral(energy, matrix, distribution):
    
    gain = np.zeros_like(distribution)
    integrand = gain.copy()
    
    for i in range(distribution.size):
        for j in range(distribution.size):
            integrand[j] = matrix[j][i]*distribution[j]
        gain[i] = integrate.trapz(integrand, x=energy)
    
    return gain

class Oxygen_Isotope_Gas:
    """Calculates the isotopic concentrations of an ideal gas given a pressure, temperature and tuple of lnXO values
    Parameters
    ============
    formula: String containing name of gas. E.G. 'CO2' or 'SO2'
    
    sym_num: Isotope symmetry number for gas. E.G. 2 for oxygen in O2 (currently only really calculates for 1 oxygen atom or 2 oxygens that have symmetry)
    
    pressure: tuple containing value of pressure and associated unit (e.g. (50., 'torr'))
    
    temperature: value of temperature (in Kelvin)
    
    conc_unit: string containing desired concentration unit (m^-3 or cm^-3)
    
    ln_values: tuple containing ln17O and ln18O values in per mil and their associated isotope standard
    """
    def __init__(self, formula, sym_num, pressure, temperature, conc_unit, ln_values):
        self.formula = formula
        self.sym_num = sym_num
        self.pressure = pressure[0]
        self.pressure_unit = pressure[1]
        self.temperature = temperature
        self.conc_unit = conc_unit
        self.ln17O = ln_values[0]
        self.ln18O = ln_values[1]
        self.isotope_std = ln_values[2]
        self.std_17O = isotope_conversion[self.isotope_std][0]
        self.std_18O = isotope_conversion[self.isotope_std][1]
        
        # Calculate pressure in pascal
        self.p_pascal = p_conversion[self.pressure_unit]*self.pressure
        
        # Calculate concentration in desired unit
        self.conc = self.p_pascal/constants.k/self.temperature/conc_conversion[self.conc_unit]
        
        # Initialize dictionary for ratios
        self.ratios = {}
        
        # Calculate isotope ratios of different species relative to all 16O containing species
        self.ratios['16O16O'] = 1.
        self.ratios['17O16O'] = self.std_17O*self.sym_num*math.exp(self.ln17O/1000)
        self.ratios['18O16O'] = self.std_18O*self.sym_num*math.exp(self.ln18O/1000)
        if self.sym_num > 1:
            self.ratios['17O17O'] = self.std_17O**self.sym_num*math.exp(self.ln17O*self.sym_num/1000)
            self.ratios['17O18O'] = self.std_17O*self.std_18O*math.exp((self.ln17O + self.ln18O)/1000)
            self.ratios['18O18O'] = self.std_18O**self.sym_num*math.exp(self.ln18O*self.sym_num/1000)
        
        # Calculate sum of ratios to find 16O16O ratio of total
        ratio_16O16O_star = 1./(math.fsum(self.ratios.values()))
        
        # Recalculate isotope ratios of relative to total instead of 16O16O
        for key in self.ratios.keys():
            self.ratios[key] *= ratio_16O16O_star

        
    def __str__(self):
        result = self.formula + ' properties:\nPressure: ' + str(self.pressure) + ' ' + self.pressure_unit + '\n'
        result += 'Pressure: ' + str(self.p_pascal) + ' Pascals\n'
        result += 'Concentration: ' + str(self.conc) + ' ' + str(self.conc_unit) + '\n'
        result += 'ln17O(' + self.name + '): ' + str(self.ln17O) + ' per mil vs.' + self.isotope_std + '\n'
        result += 'ln18O(' + self.name + '): ' + str(self.ln18O) + ' per mil vs.' + self.isotope_std + '\n'
        return result
    
    def get_name(self):
        return self.name
    
    def get_temperature(self):
        return self.temperature
    
    def get_total_conc(self):
        return self.conc, self.conc_unit
    
    def get_isotope_conc(self, isotope_ratio='16O16O'):
        """Calculates concentration of a given isotopic combination for the gas
        Parameters
        ==========
        
        isotope_ratio: a string corresponding to one of the keys in self.ratios
        
        Returns
        =======
        A tuple containing:
        A floating point containing the value of the concentration
        A string containing the unit of concentration
        A string containing the isotope ratio
        """
        if isotope_ratio in self.ratios.keys():
            result = self.ratios[isotope_ratio]*self.conc
        else:
            print("Incorrect key for calculating isotope ratio.")
            result = 0.
            
        return result, self.conc_unit, isotope_ratio
            
    def generate_gases(self, substitution=1):
        """Generates Maxwell Gas objects for each isotopic variant desired (default = 1)"""
        isotope_formula = ''
        for char in self.formula:
            if char in isotope_string.keys():
                new_char = isotope_string[char]
                isotope_formula += new_char
            elif char in string.digits:
                digit = int(char)
                isotope_formula += (digit-1)*new_char

        return isotope_formula
                
class Maxwell_Gas:
    """Contains properties and methods of a gas that follows the Maxwell-Boltzmann distribution
    Parameters
    ===========
    formula: string containing the isotopic formula of the gas
    conc: tuple containing float of concentration of gas and string for unit (Can be calculated from oxygen isotope gas or input directly)
    temperature: temperature of gas
    """    
    
    def __init__(self, formula, conc, temperature):
        self.formula = formula
        self.conc = conc[0]
        self.conc_unit = conc[1]
        self.temperature = temperature
        
        # Calculate mass using formula and isotope mass dictionary
        self.mass = 0
        for isotope in isotope_masses.keys():
            self.mass += self.formula.count(isotope) * isotope_masses[isotope]
    
    def get_mass(self):
        return self.mass
        
    def get_temperature(self):
        return self.temperature
        
    def get_conc(self):
        return self.conc, self.conc_unit
    
    def get_formula(self):
        return formula

class Collision:
    """Contains properties and methods for collisions between a Lorentz Gas and a Maxwell Gas
    maxwell_gas: Maxwell (thermal) gas object
    lorentz_gas: Lorentz (non-thermal) gas object
    elastic: elastic cross section in cm^2
    reactive: reactive cross section in cm^2 (default: 0)
    inelastic: inelastic collisions flag (default: False)
    quenching: electronic quenching collisions flag (default: False)
    isotope_exchange: cross section for isotope exchange
    kernel_calc: force calculation of kernels instead of loading from file 
    (warning: kernel calculations can be resource intensive.
    e.g. with 3.4GHz i7 intel and 8 GB of RAM, each calculation takes ~12 minutes. 
    It might be worthwhile to use CPython if many kernels need to be calculated)
    """
    
    def __init__(self, maxwell_gas, lorentz_gas, elastic, reactive = 0, inelastic = False, quenching = False, isotope_exchange = 0, kernel_calc = False):
        self.maxwell_gas = maxwell_gas
        self.lorentz_gas = lorentz_gas
        self.elastic = elastic
        self.isotope_exchange = isotope_exchange
        self.reactive = reactive
        self.new_state = self.lorentz_gas.state
        self.reactants = self.lorentz_gas.formula + oxygen_states[self.lorentz_gas.state] + ' + ' + self.maxwell_gas.formula
        
        self.kernel_calc = kernel_calc
        self.kernel = {}
        
        # Initialize and load branching ratios for different inelastic and reactive collisions
        # Call interpolation functions using the product string in branching_ratio dictionary
        self.branching_ratios = {}
        
        # Generate product string list based on keyword arguments
        
        # Initialize list and holder variables
        self.products = []
        lorentz_product = self.lorentz_gas.formula
        maxwell_product = self.maxwell_gas.formula
        
        # Stores parameters locally to avoid excess look-up
        lorentz_mass = self.lorentz_gas.get_mass()
        energy = self.lorentz_gas.energy
        maxwell_mass = self.maxwell_gas.get_mass()
        maxwell_temp = self.maxwell_gas.get_temperature()
        maxwell_conc = self.maxwell_gas.get_conc()
        mass_sqrt = 2*((lorentz_mass + maxwell_mass)*lorentz_mass)**0.5
    
        # Calculate reduced mass of collision
        self.reduced_mass = maxwell_mass*lorentz_mass/(maxwell_mass + lorentz_mass)
        
        
        #Calculate elastic kernel:
        #initializes empty 2-D array of same length as energy
        self.kernel['elastic'] = np.zeros((energy.size,energy.size))
        
        if elastic:
            
            # Calculate constants that do not depend on energy
            C = 1./100./100./4./(2*lorentz_mass*constants.k*maxwell_temp*\
                constants.value('atomic mass constant'))**0.5*\
                ((maxwell_mass/lorentz_mass)**0.5 + (lorentz_mass/maxwell_mass)**0.5)**2
            #Loop used to populate 2-D array B
            for i in range(energy.size):
                # Mass parameter vectors
                Delta_plus = (lorentz_mass/self.reduced_mass)**0.5*np.abs(energy[i]**0.5 + energy**0.5)
                Delta_minus = (lorentz_mass/self.reduced_mass)**0.5*np.abs(energy[i]**0.5 - energy**0.5)
            
                if energy[i]:
                    # Kernel calculation: from Kharchenko et al. 1998
                    self.kernel['elastic'][i,:] = np.exp((energy[i] - energy + np.abs(energy[i] - energy))/2.) *\
                        (sp.special.erf((lorentz_mass*Delta_plus + maxwell_mass*Delta_minus)/mass_sqrt) -\
                        sp.special.erf((lorentz_mass*Delta_minus + maxwell_mass*Delta_plus)/mass_sqrt) +\
                        np.exp(-np.abs(energy[i] - energy)) *\
                        (sp.special.erf((lorentz_mass*Delta_plus - maxwell_mass*Delta_minus)/mass_sqrt) -\
                        sp.special.erf((lorentz_mass*Delta_minus - maxwell_mass*Delta_plus)/mass_sqrt)))
                
                    self.kernel['elastic'][i,:] *= self.elastic/energy[i]**0.5
                else:
                    self.kernel['elastic'][i,:] = np.zeros_like(energy)
        
            self.kernel['elastic'] *= C

        
        # Finds products of isotope exchange if it occurs in the collision
        # and loads associated inelastic kernels
        if self.isotope_exchange:
            loc_lor = lorentz_gas.formula.find('O')
            isotope_lor = lorentz_gas.formula[loc_lor-2:loc_lor+1]
            loc_max = maxwell_gas.formula.find('O')
            isotope_max = maxwell_gas.formula[loc_max-2:loc_max+1]
            
            lorentz_product = lorentz_product.replace(isotope_lor, isotope_max, 1)
            maxwell_product = maxwell_product.replace(isotope_max, isotope_lor, 1)
            
            # Loads associated kernels for Lorentz gases. Note: kernel for 16O + X17O16O and X18O16O without isotope exchange not calculated
            # Because very small (<0.5%) compared to X16O16O
            if inelastic:
                if isotope_lor == '16O' and isotope_max != '16O':

                    product = lorentz_product + oxygen_states[self.lorentz_gas.state] + ' + ' + maxwell_product
                    
                    # Generate string to call file for product
                    file_string = self.reactants + '_to_' + product + '.csv'
                    file_string = file_string.replace(' ', '_')
                    file_string = file_string.replace('+', 'plus')
                    
                    self.products.append(product)
                    
                    # Find branching ratio for inelastic with isotope exchange
                    self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                                skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                                np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
                    
                    # Calculate kernel if flag set. Otherwise load from file
                    if self.kernel_calc:
                        self.kernel['exchange'] = self.calc_kernel(self.branching_ratios[product], self.isotope_exchange)
                        np.save(self.lorentz_gas.formula + self.maxwell_gas.formula + '_kernel.npy',self.kernel['exchange'])
                        print("Finished calculating kernel for " + product)
                    else:
                        self.kernel['exchange'] = np.load(self.lorentz_gas.formula + self.maxwell_gas.formula + '_kernel.npy')
                    # Note: inelastic not considered for 16O + X17O16O etc. because contribution is small
                    self.kernel['inelastic'] = np.zeros_like(self.kernel['exchange'])
                                
                    # Find branching ratio for inelastic with no isotope exchange
                    file_string = self.reactants + '_to_' + self.reactants + '.csv'
                    file_string = file_string.replace(' ', '_')
                    file_string = file_string.replace('+', 'plus')
                    
                    self.products.append(self.reactants)
                    
                    self.branching_ratios[self.reactants] = interpolate.interp1d(np.genfromtxt(file_string, 
                                skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                                np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))


                    
                    
                elif isotope_lor == isotope_max == '16O':
                    
                    product = lorentz_product + oxygen_states[self.lorentz_gas.state] + ' + ' + maxwell_product
                    
                    # Generate string to call file for each product
                    file_string = self.reactants + '_to_' + product + '.csv'
                    file_string = file_string.replace(' ', '_')
                    file_string = file_string.replace('+', 'plus')

                    self.products.append(product)
                    
                    self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                                skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                                np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
                    # Calculate kernel if flag is set. Otherwise load from file.
                    if self.kernel_calc:
                        self.kernel['inelastic'] = self.calc_kernel(self.branching_ratios[product], self.isotope_exchange)
                        np.save(self.lorentz_gas.formula + self.maxwell_gas.formula + '_kernel.npy', self.kernel['inelastic'])
                        print("Finished calculating kernel for " + product)
                    else:
                        self.kernel['inelastic'] = np.load(self.lorentz_gas.formula + self.maxwell_gas.formula + '_kernel.npy')
                        
                    self.kernel['exchange'] = np.zeros_like(self.kernel['inelastic'])
                    
                else:

                    product = lorentz_product + oxygen_states[self.lorentz_gas.state] + ' + ' + maxwell_product
                    
                    # Generate string to call file for each product
                    file_string = self.reactants + '_to_' + product + '.csv'
                    file_string = file_string.replace(' ', '_')
                    file_string = file_string.replace('+', 'plus')

                    self.products.append(product)
                    
                    self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                                skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                                np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
                                
                    # Find branching ratio for inelastic with no isotope exchange
                    file_string = self.reactants + '_to_' + self.reactants + '.csv'
                    file_string = file_string.replace(' ', '_')
                    file_string = file_string.replace('+', 'plus')
                    
                    self.products.append(self.reactants)
                    
                    self.branching_ratios[self.reactants] = interpolate.interp1d(np.genfromtxt(file_string, 
                                skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                                np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
                                
                    # Calculates kernels, otherwise loads them
                    if self.kernel_calc:
                        self.kernel['inelastic'] = self.calc_kernel(self.branching_ratios[self.reactants], self.isotope_exchange)
                        np.save(self.lorentz_gas.formula + self.maxwell_gas.formula + '_noex_kernel.npy',self.kernel['inelastic'])
                        print("Finished calculating kernel for " + self.reactants)
                        self.kernel['exchange'] = self.calc_kernel(self.branching_ratios[product], self.isotope_exchange)
                        np.save(self.lorentz_gas.formula + self.maxwell_gas.formula + '_ex_kernel.npy',self.kernel['exchange'])
                        print("Finished calculating kernel for " + product)
                    else:
                        self.kernel['inelastic'] = np.load(self.lorentz_gas.formula + self.maxwell_gas.formula + '_noex_kernel.npy')
                        self.kernel['exchange'] = np.load(self.lorentz_gas.formula + self.maxwell_gas.formula + '_ex_kernel.npy')
                             
            else:
                # Change here?
                product = lorentz_product + oxygen_states[self.lorentz_gas.state] + ' + ' + maxwell_product
                
                self.products.append(lorentz_product + oxygen_states[self.lorentz_gas.state] + ' + ' + maxwell_product)
                
                # Generate string to call file for each product
                file_string = self.reactants + '_to_' + product + '.csv'
                file_string = file_string.replace(' ', '_')
                file_string = file_string.replace('+', 'plus')

                self.products.append(product)
                    
                self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                        skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                        np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
                
                self.kernel['exchange'] = self.kernel['elastic']/self.elastic
                for i in range(energy.size):
                    self.kernel['exchange'][i,:] = self.branching_ratios[product](energy)*self.isotope_exchange

                                
        if quenching:
            # Decrements electronic state if quenching is allowed
            self.new_state -= 1
            product = self.lorentz_gas.formula + oxygen_states[self.new_state] + ' + ' + self.maxwell_gas.formula
            self.products.append(product)
            
            # Generate string to call file for each product
            file_string = self.reactants + '_to_' + product + '.csv'
            file_string = file_string.replace(' ', '_')
            file_string = file_string.replace('+', 'plus')

            self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                    skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                    np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
        elif reactive:
            # Reactive collisions are non-quenching reactions
            product = 'Products'
            self.products.append(product)
            # Generate string to call file for each product
            file_string = self.reactants + '_to_' + product + '.csv'
            file_string = file_string.replace(' ', '_')
            file_string = file_string.replace('+', 'plus')

            self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                    skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                    np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
        
        if isotope_exchange and quenching and not (isotope_lor == isotope_max == '16O'):
            product = lorentz_product + oxygen_states[self.new_state] + ' + ' + maxwell_product
            self.products.append(product)
            
            # Generate string to call file for each product
            file_string = self.reactants + '_to_' + product + '.csv'
            file_string = file_string.replace(' ', '_')
            file_string = file_string.replace('+', 'plus')

            self.branching_ratios[product] = interpolate.interp1d(np.genfromtxt(file_string, 
                    skip_header = 1, dtype = float, delimiter = ",", usecols = 0), 
                    np.genfromtxt(file_string, skip_header = 1, dtype = float, delimiter = ",", usecols = 1))
        
    def get_reactions(self):
        """Returns string of possible reaction pathways for the collision"""
        return self.reactants + ' --> ' + ', '.join(self.products)
    
    def calc_kernel(self, branch, cs):
        """Calculates kernel and stores it as a file
        Currently uses hard-sphere with c.o.m. energy branching ratio"""
        #Print initial time
        print("Starting calculation at:")
        print(time.strftime("%H:%M:%S", time.localtime()))
        
        
        #Calculate kT
        kT = constants.k*self.maxwell_gas.get_temperature()
        
        # Monte Carlo trials
        n = 25000
        
        # References energy vector
        energy = self.lorentz_gas.energy
        
        # Stores different masses locally
        reduced_mass = self.reduced_mass
        
        maxwell_mass = self.maxwell_gas.get_mass()
        
        lorentz_mass = self.lorentz_gas.get_mass()
        
        # Creates empty matrix
        B = np.empty((energy.size,energy.size))
        
        # Calculates factors that do not depend on energy UNITS: m^3/(J s)
        # (Note: concentration in m^-3 not included)
        C = cs*(lorentz_mass + maxwell_mass)**(2.5)/100./100./4.*\
            math.sqrt(2./kT/constants.value('atomic mass constant'))/maxwell_mass/lorentz_mass**2
        
        # Calculates matrix for each energy E, E'
        for i in range(energy.size):
            if not energy[i]:
                B[i,:] = 0
                continue
            #Calculates maximum and minimum energy transfer in lab energy coordinates
            Delta_minus = (lorentz_mass/reduced_mass)*(energy[i]**0.5 - energy**0.5)**2
            Delta_plus = (lorentz_mass/reduced_mass)*(energy[i]**0.5 + energy**0.5)**2
            for j in range(energy.size):
                #Randomly generate E_r (reduced c.o.m. energy) and Q for monte carlo calculation
                E_r = np.random.uniform(0,energy[-1],n)
                Q = np.random.uniform(Delta_minus[j],Delta_plus[j],n)
                #Calculate integrand samples
                integrand_samples = np.exp(-E_r*(1 + maxwell_mass/lorentz_mass) + maxwell_mass/2./lorentz_mass*Q)*branch(E_r)/2./Q**0.5*\
                                special.i0(maxwell_mass/2./lorentz_mass/Q*np.sqrt(Q*(4*E_r - Q)*(Delta_plus[j] - Q)*(Q - Delta_minus[j])))
                #Calculates integrand value
                B[i,j] = bn.nansum(integrand_samples)*(Delta_plus[j] - Delta_minus[j])*energy[-1]/float(n)
            B[i,:] *= np.exp((energy[i] - energy)/2. - maxwell_mass/lorentz_mass/2.*(energy + energy[i]))/energy[i]**0.5
            B[i,:] = smooth(B[i,:],window_len=20)

        print("Ending calculation at:")
        print(time.strftime("%H:%M:%S", time.localtime()))
        return B*C
            
        


class Lorentz_Gas(Maxwell_Gas):
    """Contains properties and methods for calculating an energy distribution of a Lorentz Gas
    Derived from Maxwell_Gas
    Additional Parameters
    ======================
    state: number containing electronic state of gas (0 = ground state)
    
    rate: tuple containing the rate of formation of the Lorentz Gas and the associated unit string for concentration in rate (per s automatically added)
    
    energy: energy vector used for calculating distribution
    
    initial_dist_file: flag used to find initial distribution from file and map onto energy space
                       defaults to Maxwell-Boltzmann otherwise
    
    Note: temperature not set since gas is non-thermal
    """
    def __init__(self, formula, state, rate, energy, initial_dist_file=True,maxwell_rate = (0,'cm^-3 s^-1')):
        Maxwell_Gas.__init__(self, formula, rate, None)
        self.state = state
        self.energy = energy
        self.rate = rate[0]
        self.rate_unit = rate[1]
        self.maxwell_rate = maxwell_rate[0]
        self.maxwell_rate_unit = maxwell_rate[1]
        
        # Load initial distribution file
        if initial_dist_file:
            # Load distribution from file based off name of Lorentz Gas
            file_string = 'norm_' + self.formula + oxygen_states[self.state] + '.csv'
            x_L = np.genfromtxt(file_string, dtype = float, delimiter = ",", usecols = 0)
            dist = np.genfromtxt(file_string, dtype = float, delimiter = ",", usecols = 1)
            
            # Create interpolation function for distribution
            dist_f = interpolate.interp1d(x_L, dist)
            
            # Create initial distribution vector from interpolation function and energy space
            self.initial_dist = dist_f(energy)*rate[0]
        else:
            self.initial_dist = np.zeros_like(energy)
            #self.initial_dist = maxwell_boltzmann(energy)*rate[0]
        
        self.initial_dist += maxwell_boltzmann(energy)*maxwell_rate[0]
        self.exchange_source = np.zeros_like(self.initial_dist)
        self.distribution = self.initial_dist.copy()
        self.coll_loss = np.ones_like(self.initial_dist)
        self.chem_loss = np.ones_like(self.initial_dist)
        self.collisions = {}
        self.rate_constants = {}
    
    def generate_collisions(self, maxwell_gases, elastic, reactive = 0, inelastic = False, quenching = False, isotope_exchange = 0, kernel_calc = False):
        """Generates collisions from LIST of Maxwell Gas objects"""
        for gas in maxwell_gases:
            self.collisions[gas.formula] = Collision(gas, self, elastic, reactive, inelastic, quenching, isotope_exchange, kernel_calc)
    
    def calculate_distribution(self):
        """Calculates the total kernel from all energy transfer collisions with a maxwell gas and the concentration of each species.
        Calculates a total kernel from all non-quenching isotope exchange collisions with a maxwell gas.
        Calculates chemical loss and collisional loss vectors.
        Then, calculates energy distribution by iteration.
        Must generate collisions first using generate collisions method."""
        
        # Initialize matrices
        self.total_kernel = np.zeros((self.energy.size, self.energy.size))
        self.exchange_kernel = np.zeros((self.energy.size, self.energy.size))
        
        # Iteratre over each possible collision
        for collision in self.collisions.values():
            # Calculate concentration of associated maxwell gas
            conc = collision.maxwell_gas.get_conc()[0]
            self.kernel_temp = collision.maxwell_gas.get_temperature()
            
            # Store in exchange kernel or total kernel
            for name in collision.kernel.keys():
                if name == 'exchange':
                    self.exchange_kernel += collision.kernel[name]*conc*100**3
                else:
                    self.total_kernel += collision.kernel[name]*conc*100**3
                    
        # Calculate chemical loss            
        # Initialize vector
        self.chem_loss = np.zeros(self.energy.size)
        
        # Store mass of lorentz gas locally
        lorentz_mass = self.mass
        # Iterate over every possible collision
        for collision in self.collisions.values():
            # Store mass of maxwell gas and reduced mass locally
            maxwell_mass = collision.maxwell_gas.get_mass()
            reduced_mass = collision.reduced_mass
            temp = collision.maxwell_gas.get_temperature()
            conc = collision.maxwell_gas.get_conc()[0]
            # Iterate over each possible product
            for product in collision.products:
                # (Re-)initialize temporary loss vector
                loss = np.zeros(self.energy.size)
                # Check if actually a reactive collision
                if not oxygen_states[self.state] in product:
                    for i in range(self.energy.size):
                        # Avoid dividing by zero
                        if energy[i]:
                            # Calculates c.o.m. energy distribution for particle of lab frame energy E
                            COM_dist = ((lorentz_mass + maxwell_mass)/math.pi/energy[i]/reduced_mass)**0.5 *\
                                np.sinh(2*maxwell_mass/lorentz_mass*(energy[i]*energy*lorentz_mass/reduced_mass)**0.5) *\
                                np.exp(-energy*maxwell_mass/reduced_mass - energy[i]*maxwell_mass/lorentz_mass)
                            
                            # Calculate reaction rate for each c.o.m. energy
                            COM_dist *= (2*energy*constants.k*temp/reduced_mass/constants.value('atomic mass constant'))**0.5
                            COM_dist *= (collision.branching_ratios[product](energy)*collision.reactive)*conc*100.
                            
                            # Integrate to get total loss rate at a given lab frame energy E
                            loss[i] = integrate.trapz(COM_dist,x=energy)
                            
                    # Prevent dividing by zero
                    loss[0] = loss[1]
                    
                # Add loss term for given collision to total chemical loss
                self.chem_loss += loss
                
        # Calculate collisional loss
        # Initialize vector
        self.coll_loss = np.zeros_like(self.energy)
        
        # Sum of energy transfer and exchange kernels
        temp_kernel = self.total_kernel + self.exchange_kernel
        
        # Integrate at each energy in the kernel matrix
        for i in range(self.energy.size):
            self.coll_loss[i] = integrate.trapz(temp_kernel[:][i], x=self.energy)
        
        # Apply chain rule
        self.coll_loss *= constants.k*self.kernel_temp
        
        # Calculate kinetic energy distribution
        # Iteration counters
        o = 0
        flag = 1
        
        # Initialize collision gain vector
        self.coll_gain = np.zeros_like(self.energy)
        
        
        # Iterate until flag is 0
        while flag:
            flag = 0
            
            gain = np.zeros_like(self.energy)
            # Calculate collisional gain vector for current distribution
            self.coll_gain = collision_gain_integral(self.energy, self.total_kernel, self.distribution)*constants.k*self.kernel_temp
            
            # Calculate new distribution from gain and loss terms
            new_distribution = (self.initial_dist + self.exchange_source + self.coll_gain)/(self.coll_loss + self.chem_loss)
            
            # Check how different new distribution is from old distribution
            for i in range(self.distribution.size):
                if new_distribution[i]:
                    if math.fabs(self.distribution[i]/new_distribution[i] - 1) > 0.01: flag += 1
            
            # Copy new distribution into distribution
            self.distribution = new_distribution.copy()
            
            # Prevent infinite loop
            o += 1
            if o > 300:
                print("exceeded iteration limit")
                break
        
        print(o, flag)
        
        # Find concentration from integral of distribution
        self.conc = integrate.trapz(self.distribution, x=self.energy)
        self.conc_unit = self.rate_unit
        
        # Find normalized distribution
        self.distribution_norm = self.distribution/self.conc
        
    def calculate_exchange(self, other_dist, collision):
        """Calculates second exchange source from distirubtion and collision kernel. 
        E.G. source term for 16O(1D) + 12C17O16O --> 17O(1D) + 12C16O16O"""
        new_kernel = collision.kernel['exchange']*collision.maxwell_gas.get_conc()[0]*100**3
        self.exchange_source = collision_gain_integral(self.energy, new_kernel, other_dist)*constants.k*collision.maxwell_gas.get_temperature()
        
    def calculate_rate_constants(self):
        """Calculates rate constants from distribution"""
        # Store mass of lorentz gas locally
        lorentz_mass = self.mass
        # Iterate over every possible collision
        for collision in self.collisions.values():
            # Store mass of maxwell gas and reduced mass locally
            maxwell_mass = collision.maxwell_gas.get_mass()
            reduced_mass = collision.reduced_mass
            temp = collision.maxwell_gas.get_temperature()
            # Iterate over each possible product
            for product in collision.products:
                COM_dist = np.zeros(self.energy.size)
                # (Re-)initialize temporary integrand vector
                integrand = np.zeros(self.energy.size)
                for i in range(self.energy.size):
                    for j in range(self.energy.size):
                        # Avoid dividing by zero
                        if energy[j]:
                        # Calculates integrand for c.o.m. energy distribution for particle of lab frame energy E
                            integrand[j] = self.distribution[j]*((lorentz_mass + maxwell_mass)/math.pi/energy[j]/reduced_mass)**0.5 *\
                            np.sinh(2*maxwell_mass/lorentz_mass*(energy[i]*energy[j]*lorentz_mass/reduced_mass)**0.5) *\
                            np.exp(-energy[i]*maxwell_mass/reduced_mass - energy[j]*maxwell_mass/lorentz_mass)
                    # Find COM distribution at point energy[i]    
                    COM_dist[i] = integrate.trapz(integrand, x=energy)
                
                # Normalize distribution
                COM_dist = COM_dist/integrate.trapz(COM_dist, x=energy)
                # Calculate reaction rate for each c.o.m. energy
                COM_dist *= (2*energy*constants.k*temp/reduced_mass/constants.value('atomic mass constant'))**0.5
                COM_dist *= (collision.branching_ratios[product](energy)*collision.reactive)*100.
                self.rate_constants[collision.reactants + ' --> ' + product] = integrate.trapz(COM_dist, x=energy)
                
    def write_dist_to_file(self, filename):
        """Writes photolysis, exchange, total initial, and final energy distributions to file filename"""
        
        # Normalize photolysis distribution        
        temp_norm = integrate.trapz(self.initial_dist, x=self.energy)
        photolysis_norm = self.initial_dist/temp_norm        
        
        # Normalize exchange distribution
        temp_norm = integrate.trapz(self.exchange_source, x=self.energy)
        exchange_norm = self.exchange_source/temp_norm
        
        # Normalize total initial distribution
        total_init = self.exchange_source + self.initial_dist
        temp_norm = integrate.trapz(total_init, x=self.energy)
        total_norm = (total_init)/temp_norm
        
        output = np.array((self.energy, self.initial_dist, photolysis_norm, self.exchange_source, exchange_norm, total_init, total_norm, self.distribution, self.distribution_norm), dtype=float)
        output = np.swapaxes(output, 0, 1)
        with open(filename, 'ba') as dist_file:
            np.savetxt(dist_file, output, delimiter=',', fmt='%.8e', \
                       header=self.formula + oxygen_states[self.state] + '\n'+\
                           'Energy (unitless), Photolysis Distribution (cm^-3 J^-1),  Normalized (J^-1), '+\
                           'Exchange Distribution (cm^-3 J^-1), Normalized (J^-1), '+\
                           'Total Initial Distribution (cm^-3 J^-1), Normalized (J^-1), '+\
                           'Calculated Distribution (cm^-3 J^-1), Normalized (J^-1)',\
                       footer=' , Concentration:, {0:6E}'.format(self.conc))
            
    
    def write_rate_k_to_file(self, filename):
        """Writes rate constants to file filename"""
        with open(filename, 'w') as rate_file:
            for reaction in self.rate_constants.keys():
                rate_file.write(reaction+ ', {0:8E}'.format(self.rate_constants[reaction]) + '\n')
            
    
# Initialize temperature
Temp = 298.

#Create named tuple describing experimental conditions:
#name: Text string describing conditions to be placed in file header
#pressure (p,Torr), O2/CO2 mixing ratio (rho,unitless), O3 number density (n_O3, cm^-3).
#Isotopic compositions of O2 and CO2 and steady state O1D formation rates for each isotope
#calculated using KINTECUS model 33 (o3co2_33_062211.xls)
Conditions = namedtuple('Conditions', ['name', 'rho', 'p', 'T', 'n_O3', 
'ln_values_O2', 'ln_values_CO2'])

# Initialize conditions dictionary
initial_cond = {}


# Initialize rate dictionary (calculated from KINTECUS model 33)
initial_rates = {(1000, '16O') : (2.03308E13, 'cm^-3'),
                (1000, '17O') : (9.25369E9, 'cm^-3'),
                (1000, '18O') : (4.76637E10, 'cm^-3'),
                (10, '16O') : (2.04232E13, 'cm^-3'),
                (10, '17O') : (9.14951E9, 'cm^-3'),
                (10, '18O') : (4.71781E10, 'cm^-3'),
                (1, '16O') : (1.45992E13, 'cm^-3'),
                (1, '17O') : (6.10822E9, 'cm^-3'),
                (1, '18O') : (3.16416E10, 'cm^-3'),
                (0.1, '16O') : (3.18971E12, 'cm^-3'),
                (0.1, '17O') : (1.25459E9, 'cm^-3'),
                (0.1, '18O') : (6.52546E9, 'cm^-3'),
                (0.01, '16O') : (3.74292E11, 'cm^-3'),
                (0.01, '17O') : (1.45302E8, 'cm^-3'),
                (0.01, '18O') : (7.56357E8, 'cm^-3')}

# Initial parameters for calculations
initial_cond['rho1000'] = Conditions(name='p = 50 rho = 1000', rho=1000., p=(50., 'torr'), T=Temp,
    n_O3 = (1.24833E15, 'cm^-3'), ln_values_O2 = (12.48, 26.01, 'VSMOW'), ln_values_CO2 = (162.94, 156.31, 'VSMOW'))
    
initial_cond['rho10'] = Conditions(name='p = 50 rho = 10', rho=10., p=(50., 'torr'), T=Temp,
    n_O3 = (1.25399E15, 'cm^-3'), ln_values_O2 = (-2.89, 11.89, 'VSMOW'), ln_values_CO2 = (147.09, 141.54, 'VSMOW'))

initial_cond['rho1'] = Conditions(name='p = 50 rho = 1', rho=1., p=(50., 'torr'), T=Temp,
    n_O3 = (8.96761E14, 'cm^-3'), ln_values_O2 = (-69.14, -49.16, 'VSMOW'), ln_values_CO2 = (78.73, 77.78, 'VSMOW'))

initial_cond['rho01'] = Conditions(name='p = 50 rho = 0.1', rho=0.1, p=(50., 'torr'), T=Temp,
    n_O3 = (1.96896E14, 'cm^-3'), ln_values_O2 = (-130.25, -105.61, 'VSMOW'), ln_values_CO2 = (15.83, 18.92, 'VSMOW'))

initial_cond['rho001'] = Conditions(name='p = 50 rho = 0.01', rho=0.01, p=(50., 'torr'), T=Temp,
    n_O3 = (2.29892E13, 'cm^-3'), ln_values_O2 = (-141.8, -116.3, 'VSMOW'), ln_values_CO2 = (4.19, 7.83, 'VSMOW'))
    
        
#Generates x space for reduced energy (x = E/kT).
energy = np.linspace(0,(length-1)*scale,length)

# cross sections for o2
o2_reactive = 4.07025E-16
o2_elastic = 2.8E-15

#Reactive cross section for o3
o3_reactive = 3.3E-15

#Total O(1D) + CO2 cross section
co2_total_cs = 3.5170E-15
#Combined inelastic and reactive cross section
co2_rxnin_cs = 1.4953E-15
# Elastic cross section
co2_elastic = co2_total_cs - co2_rxnin_cs

# Kernel calculation flag (Leave set to false unless you have time to spare)
force_kernel_calc = False

for condition in initial_cond.values():
    
    p_O2 = (condition.p[0]/(1+1/condition.rho), condition.p[1])
    p_CO2 = (condition.p[0]/(1+condition.rho), condition.p[1])

    O2_calc = Oxygen_Isotope_Gas('O2', 2, p_O2, Temp, 'cm^-3', condition.ln_values_O2)
    CO2_calc = Oxygen_Isotope_Gas('CO2', 2, p_CO2, Temp, 'cm^-3', condition.ln_values_CO2)


    # Initialize dictionaries
    O1D = {}
    O2 = {}
    CO2 = {}
    
    # Initialize isotopomers of O1D
    for isotope in oxygen_isotopes:
        O1D[isotope] = Lorentz_Gas(isotope, 1, initial_rates[(condition.rho, isotope)], energy)
        pass
    
    #O1D['16O'] = Lorentz_Gas('16O', 1, initial_rates[(condition.rho, isotope)], energy)
    #O1D['17O'] = Lorentz_Gas('17O', 1, initial_rates[(condition.rho, isotope)], energy, initial_dist_file=False)
    #O1D['18O'] = Lorentz_Gas('18O', 1, initial_rates[(condition.rho, isotope)], energy, initial_dist_file=False)

        
    # Initialize O3
    O3 = [Maxwell_Gas('16O16O16O', condition.n_O3, O2_calc.get_temperature())]

    # Initialize isotopomers of O2 and CO2
    for isotopomer in O2_combinations:
        O2[isotopomer] = Maxwell_Gas(isotopomer, O2_calc.get_isotope_conc(isotopomer), O2_calc.get_temperature())
        CO2['12C'+isotopomer] = Maxwell_Gas('12C'+isotopomer, CO2_calc.get_isotope_conc(isotopomer), CO2_calc.get_temperature())

    # Generate collision data between O1D isotopomers and each Maxwell gas

    O1D['16O'].generate_collisions(O2.values(), o2_elastic, reactive = o2_reactive, quenching = True)
    O1D['16O'].generate_collisions(O3, 0, reactive = o3_reactive)
    O1D['16O'].generate_collisions(CO2.values(), co2_elastic, reactive = co2_rxnin_cs, inelastic = True, quenching = True, isotope_exchange = co2_rxnin_cs) 
    
    for isotope in rare_oxygen_isotopes:
        O1D[isotope].generate_collisions([O2['16O16O']], o2_elastic, reactive = o2_reactive, quenching = True)
        O1D[isotope].generate_collisions(O3, 0, reactive = o3_reactive)
        O1D[isotope].generate_collisions([CO2['12C16O16O']], co2_elastic, reactive = co2_rxnin_cs, inelastic = True, quenching = True, isotope_exchange = co2_rxnin_cs, kernel_calc = force_kernel_calc)
    
    if force_kernel_calc:
        force_kernel_calc = False
    
    # Calculate energy distribution for 16O1D
    O1D['16O'].calculate_distribution()

    # Calculate rate constants for 16O1D
    O1D['16O'].calculate_rate_constants()

    # Calculate exchange source term for 16O1D + 45CO2 --> 17O1D + 44CO2
    O1D['17O'].calculate_exchange(O1D['16O'].distribution, O1D['16O'].collisions['12C17O16O'])

    # Calculate energy distribution for 17O1D
    O1D['17O'].calculate_distribution()

    # Calculate rate constants for 17O1D
    O1D['17O'].calculate_rate_constants()

    # Calculate exchange source term for 16O1D + 46CO2 --> 18O1D + 44CO2
    O1D['18O'].calculate_exchange(O1D['16O'].distribution, O1D['16O'].collisions['12C18O16O'])

    # Calculate energy distribution for 18O1D
    O1D['18O'].calculate_distribution()

    # Calculate rate constants for 18O1D
    O1D['18O'].calculate_rate_constants()


    #~ for isotope in oxygen_isotopes:
        #~ with open(condition.name + ' ' + isotope + '.csv', 'w') as object_file:
            #~ cPickle.dump(O1D[isotope], object_file)

    for isotope in oxygen_isotopes:
        file = condition.name+ ' ' + isotope + ' ' + str(datetime.date.today())+ '.csv'
        O1D[isotope].write_dist_to_file('dist '+file)
        O1D[isotope].write_rate_k_to_file('rate k '+file)
            
    legend = ['16O', '17O', '18O', 'maxwell']

    plt.plot(energy, O1D['16O'].distribution_norm, energy, O1D['17O'].distribution_norm, energy, O1D['18O'].distribution_norm, energy, maxwell_boltzmann(energy))
    plt.legend(legend)
    plt.show()
