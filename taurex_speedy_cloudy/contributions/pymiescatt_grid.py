from taurex.contributions.contribution import Contribution
import numpy as np
from taurex.data.fittable import fitparam
import numba
import scipy.stats as stats
import h5py
from taurex.exceptions import InvalidModelException
from taurex.util.util import create_grid_res


@numba.jit(nopython=True, nogil=True)
def contribute_mie_tau(startK, endK, sigma, path, ngrid, layer, tau):
    for k in range(startK, endK):
        _path = path[k]
        for wn in range(ngrid):
            tau[layer, wn] += sigma[k+layer, wn]*_path

class InvalidPyMieScattGridException(InvalidModelException):
    """
    Exception that is called when the contributio fails
    """
    pass

class PyMieScattGridExtinctionContribution(Contribution):
    """
    Computes Mie scattering contribution to optical depth
    using the Bohren and Huffmann in its PyMieScatt implementation.

    Parameters
    ----------
    mie_particle_mean_radius: Mean radius of the particles in um
    mie_particle_mix_ratio: Number density in molecules/m^3 --> Divide this number by 1,000,000 to get this in more common molecules/cm^3
    mie_midP: Middle of the clouds in Pa
    mie_rangeP: Extend of the clouds in log scale. If mie_midP = 1e5 Pa and mie_rangeP = 1 then clouds extend from 1e6 to 1e4 Pa
    """

    def __init__(self, mie_particle_mean_radius=[0.01,], 
                 mie_particle_logstd_radius = [0.001], ## Serves for the normaly distributed particle size distribution.
                 mie_particle_paramA = [1., ], mie_particle_paramB = [6.,], mie_particle_paramC = [6.,], mie_particle_paramD = [1.,], ## Serves for Deirmendjian particle size distribution.
                 mie_particle_radius_Nsampling = 5, mie_particle_radius_Dsampling = 2, ## Is used for sampling the particle distribution.
                 mie_particle_radius_distribution = 'normal', ## choices are 'normal', 'budaj', 'deirmendjian'.
                 mie_species_path=None, species = ['Mg2SiO4'], file_extension = '.h5',
                 mie_particle_mix_ratio=[1e-10], 
                 mie_porosity = None,
                 mie_midP = [1e3],
                 mie_rangeP = [1],
                 mie_nMedium=1, 
                 mie_resolution = 100,
                 mie_particle_altitude_distrib = 'exp_decay',
                 mie_particle_altitude_decay = [-5], ## Controls the decay rate, inspired by Whitten et al. 2008 / Attreya et al. 2005
                 name = 'PyMieScattGridExtinction'):
        super().__init__(name)

        self._mie_particle_mean_radius = mie_particle_mean_radius
        self._mie_particle_std_radius = mie_particle_logstd_radius
        self._mie_particle_paramA = mie_particle_paramA
        self._mie_particle_paramB = mie_particle_paramB
        self._mie_particle_paramC = mie_particle_paramC
        self._mie_particle_paramD = mie_particle_paramD

        if mie_particle_radius_distribution == 'deirmendjian':
            self._mie_particle_mean_radius = None
            self.warning('The Rmean parameter is being disabled because not needed for Deirmendjian 1964 particle distribution')
        else:
            self._mie_particle_paramA = None
            self.warning('The mie_particle_paramA parameter is being disabled because only used in Deirmendjian 1964 particle distribution')

        if mie_particle_radius_distribution == 'budaj':
            self._mie_particle_std_radius = None
            self.warning('The Rlogstd parameter is being disabled because not needed for Bujaj 2015 particle distribution')       

        self._mie_particle_radius_distribution = mie_particle_radius_distribution

        self._mie_particle_mix_ratio = mie_particle_mix_ratio
        self._mie_porosity = mie_porosity

        self._mie_midP = mie_midP
        self._mie_rangeP = mie_rangeP

        self._Nsampling = int(mie_particle_radius_Nsampling)
        self._Dsampling = mie_particle_radius_Dsampling

        self._mie_species_path = mie_species_path
        self._species = species

        self._particle_alt_distib = mie_particle_altitude_distrib
        self._particle_alt_decay = mie_particle_altitude_decay
        self._file_extension = file_extension
        self._mie_nMedium = mie_nMedium

        self._resolution = mie_resolution
        self._radius_grid, self._Qext, self._Qext_wn,  _ = self.load_input_files(self._mie_species_path, 
                                                self._species)
        
        self.generate_particle_fitting_params()

    def load_input_files(self, path, species, extension = '.h5'):
        """_summary_

        Args:
            path (str): _description_
            species (str): _description_
            extension (str): _description_

        Returns:
            np.array: radius grid
            np.array: Qext grid
            np.array: paths to the files
            
        Loads the input files for the PyMieScatt contribution Qext and matching radius. only work for a single molecule at the moment.
        """
        paths = []
        radius_grids, Qexts, wavenumber_grids = [], [], []
        for specie in species:  
            file_path = path+'/'+specie+extension
            Qext_grid = h5py.File(file_path)

            paths.append(file_path)
            
            #wls = 1e4 / Qext_grid["wavenumber_grid"][()]
            #order = np.argsort(wls)

            radius_grids.append( Qext_grid["radius_grid"][()] )
            Qexts.append( Qext_grid["Qext"][()] )
            wavenumber_grids.append( Qext_grid["wavenumber_grid"][()] )

            Qext_grid.close()
            
        return radius_grids, Qexts, wavenumber_grids, paths

    def contribute(self, model, start_layer, end_layer,
                   density_offset, layer, density, tau, path_length=None):

        contribute_mie_tau(start_layer, end_layer, self.sigma_xsec, path_length, self._ngrid, layer, tau)

    def generate_particle_fitting_params(self):

        bounds_Rm = [0.01, 10]
        bounds_Rstd = [0.01, 0.2]
        bounds_X = [1e0, 1e12]
        bounds_midP = [1e6, 1e0]
        bounds_rangeP = [0.0, 3]
        bounds_decayP = [-7, 0]
        bounds_poro = [0,1]

        ### CREATE JOINED FITPARAMS
        if self._mie_particle_mean_radius is not None:
            param_name = 'Rmean_share'
            param_latex = '$Rmean_share$'
            def read_RmeanShare(self):
                return np.mean(self._mie_particle_mean_radius)
            def write_RmeanShare(self, value):
                self._mie_particle_mean_radius[:] = [value]*len(self._mie_particle_mean_radius)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, read_RmeanShare,
                                    write_RmeanShare, 'log', default_fit, bounds_Rm)
        if self._mie_particle_std_radius is not None:
            param_name = 'Rlogstd_share'
            param_latex = '$Rlogstd_share$'
            def read_RstdShare(self):
                return np.mean(self._mie_particle_std_radius)
            def write_RstdShare(self, value):
                self._mie_particle_std_radius[:] = [value]*len(self._mie_particle_std_radius)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, read_RstdShare,
                                    write_RstdShare, 'linear', default_fit, bounds_Rstd)
            
        param_name = 'X_share'
        param_latex = '$X_share$'
        def read_XShare(self):
            return np.mean(self._mie_particle_mix_ratio)
        def write_XShare(self, value):
            self._mie_particle_mix_ratio = [value]*len(self._mie_particle_mix_ratio)
        default_fit = False
        self.add_fittable_param(param_name, param_latex, read_XShare,
                                write_XShare, 'log', default_fit, bounds_X)

        param_name = 'midP_share'
        param_latex = '$midP_share$'
        def read_midPShare(self):
            return np.mean(self._mie_midP)
        def write_midPShare(self, value):
            self._mie_midP[:] = [value]*len(self._mie_midP)
        default_fit = False
        self.add_fittable_param(param_name, param_latex, read_midPShare,
                                write_midPShare, 'log', default_fit, bounds_midP)

        param_name = 'rangeP_share'
        param_latex = '$rangeP_share$'
        def read_rangePShare(self):
            return np.mean(self._mie_rangeP)
        def write_rangePShare(self, value):
            self._mie_rangeP[:] = [value]*len(self._mie_rangeP)
        default_fit = False
        self.add_fittable_param(param_name, param_latex, read_rangePShare,
                                write_rangePShare, 'linear', default_fit, bounds_rangeP)
        
        param_name = 'decayP_share'
        param_latex = '$decayP_share$'
        def read_decayPShare(self):
            return np.mean(self._particle_alt_decay)
        def write_decayPShare(self, value):
            self._particle_alt_decay[:] = [value]*len(self._particle_alt_decay)
        default_fit = False
        self.add_fittable_param(param_name, param_latex, read_decayPShare,
                                write_decayPShare, 'linear', default_fit, bounds_decayP)

        ### CREATE INDIVIDUAL SPECIES FITPARAMS
        for idx, val in enumerate(self._species):
            
            if self._mie_particle_mean_radius is not None:
                param_name = 'Rmean_{}'.format(val)
                param_latex = '$Rmean_{}$'.format(val)
                def read_Rmean(self, idx=idx):
                    return self._mie_particle_mean_radius[idx]
                def write_Rmean(self, value, idx=idx):
                    self._mie_particle_mean_radius[idx] = value
                default_fit = False
                self.add_fittable_param(param_name, param_latex, read_Rmean,
                                        write_Rmean, 'log', default_fit, bounds_Rm)

            if self._mie_particle_std_radius is not None:
                param_name = 'Rlogstd_{}'.format(val)
                param_latex = '$Rlogstd_{}$'.format(val)
                def read_Rstd(self, idx=idx):
                    return self._mie_particle_std_radius[idx]
                def write_Rstd(self, value, idx=idx):
                    self._mie_particle_std_radius[idx] = value
                default_fit = False
                self.add_fittable_param(param_name, param_latex, read_Rstd,
                                        write_Rstd, 'linear', default_fit, bounds_Rstd)
                
            if self._mie_porosity is not None:
                param_name = 'Porosity_{}'.format(val)
                param_latex = '$Porosity_{}$'.format(val)
                def read_Poro(self, idx=idx):
                    return self._mie_porosity[idx]
                def write_Poro(self, value, idx=idx):
                    self._mie_porosity[idx] = value
                default_fit = False
                self.add_fittable_param(param_name, param_latex, read_Poro,
                                        write_Poro, 'linear', default_fit, bounds_poro)

            param_name = 'X_{}'.format(val)
            param_latex = '$X_{}$'.format(val)
            def read_X(self, idx=idx):
                return self._mie_particle_mix_ratio[idx]
            def write_X(self, value, idx=idx):
                self._mie_particle_mix_ratio[idx] = value
            default_fit = False
            self.add_fittable_param(param_name, param_latex, read_X,
                                    write_X, 'log', default_fit, bounds_X)

            param_name = 'midP_{}'.format(val)
            param_latex = '$midP_{}$'.format(val)
            def read_midP(self, idx=idx):
                return self._mie_midP[idx]
            def write_midP(self, value, idx=idx):
                self._mie_midP[idx] = value
            default_fit = False
            self.add_fittable_param(param_name, param_latex, read_midP,
                                    write_midP, 'log', default_fit, bounds_midP)

            param_name = 'rangeP_{}'.format(val)
            param_latex = '$rangeP_{}$'.format(val)
            def read_rangeP(self, idx=idx):
                return self._mie_rangeP[idx]
            def write_rangeP(self, value, idx=idx):
                self._mie_rangeP[idx] = value
            default_fit = False
            self.add_fittable_param(param_name, param_latex, read_rangeP,
                                    write_rangeP, 'linear', default_fit, bounds_rangeP)
            
            param_name = 'decayP_{}'.format(val)
            param_latex = '$decayP_{}$'.format(val)
            def read_decayP(self, idx=idx):
                return self._particle_alt_decay[idx]
            def write_decayP(self, value, idx=idx):
                self._particle_alt_decay[idx] = value
            default_fit = False
            self.add_fittable_param(param_name, param_latex, read_decayP,
                                    write_decayP, 'linear', default_fit, bounds_decayP)

    
    def prepare_each(self, model, wngrid):

        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_profile = model.pressureProfile
        
        sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))


        for specie_idx, s in enumerate(self._species):
            wn = self._Qext_wn[specie_idx]
            Rmean = self._mie_particle_mean_radius[specie_idx]
        
            
            ## GET A LOG DISTRIBUTION OF THE PARTICLE RADIUS

            if self._mie_particle_radius_distribution == 'budaj': ## This distribution can be found in Budaj et al. 2015
                LogRsigma = 0.2 ## since the distribution is fixed in width, this can be set to approx 0.2 for the sampling
                radii_log = np.linspace(10**(np.log10(Rmean)+self._Dsampling*LogRsigma), 10**(np.log10(Rmean)-self._Dsampling*LogRsigma), self._Nsampling)
                weights = ((radii_log/Rmean)**6)*np.exp(-6*radii_log/Rmean)
            elif self._mie_particle_radius_distribution == 'deirmendjian': ## This distribution can be found in Deirmendjian 1964 (modified Gamma distribution)
                LogRsigma = self._mie_particle_std_radius[specie_idx]
                radii_log = np.linspace(10**(np.log10(Rmean)+self._Dsampling*LogRsigma), 10**(np.log10(Rmean)-self._Dsampling*LogRsigma), self._Nsampling)
                weights = self._mie_particle_paramA[specie_idx]*(radii_log**self._mie_particle_paramB[specie_idx])*np.exp(-self._mie_particle_paramC[specie_idx]*(radii_log**self._mie_particle_paramD[specie_idx]))
            else: ## This is simply a normal distribution.
                LogRsigma = self._mie_particle_std_radius[specie_idx]
                radii_log = np.linspace(10**(np.log10(Rmean)+self._Dsampling*LogRsigma), 10**(np.log10(Rmean)-self._Dsampling*LogRsigma), self._Nsampling)
                weights = stats.norm.pdf(np.log10(radii_log), np.log10(Rmean), LogRsigma)
            Qexts = []
            
            for radius in radii_log:
                closest_indice = np.abs(self._radius_grid[specie_idx] - radius).argmin()
                if radius > self._radius_grid[specie_idx][closest_indice]:
                    radii_neighbor_idx = [closest_indice, closest_indice + 1]
                else:
                    radii_neighbor_idx = [closest_indice - 1, closest_indice]
                    
                R1 = self._radius_grid[specie_idx][radii_neighbor_idx[0]]
                R2 = self._radius_grid[specie_idx][radii_neighbor_idx[1]]
                delta_R = R1 - R2
                
                Q1_arr = self._Qext[specie_idx][radii_neighbor_idx[0]]  # shape: (n_wavelengths,)
                Q2_arr = self._Qext[specie_idx][radii_neighbor_idx[1]]

                a = (Q1_arr - Q2_arr) / delta_R
                b = Q1_arr - a * R1

                # Interpolate at current log radius
                Qexts.append(a * radius + b)
                                
            
            Qexts =  np.array(Qexts) * np.power(radii_log[:, np.newaxis] * 1e3, 2)   # As Qext was coomputed with radii in mm the radii here also needs to be in mm not um.
            
            
            Qext_mean = np.average(Qexts, axis=0, weights=weights)
            Qext_int = np.interp(wngrid, wn[::-1], Qext_mean, left=0, right=0)
            Qext_int = Qext_int[::-1]
            sigma_mie = np.zeros((len(wngrid)))
            


            sigma_mie[Qext_int!=0] = Qext_int[Qext_int!=0]* np.pi * 1e-18
            ## So here sigma_mie is in m2 (nm2 to m2 conversion is 1e-18)
            


            if self._mie_midP[specie_idx] == -1:
                bottom_pressure = pressure_profile[0]
                top_pressure = pressure_profile[-1]
            else:
                bottom_pressure = 10**(np.log10(self._mie_midP[specie_idx]) + self._mie_rangeP[specie_idx]/2)
                top_pressure = 10**(np.log10(self._mie_midP[specie_idx]) - self._mie_rangeP[specie_idx]/2)

            cloud_filter = (pressure_profile <= bottom_pressure) & \
                (pressure_profile >= top_pressure)

            sigma_xsec_int = np.zeros(shape=(self._nlayers, wngrid.shape[0]))
            
            ## This line implied that self._mie_particle_mix_ratio is expressed in m-3
            if self._particle_alt_distib == 'exp_decay':
                ## if we want it with exp decay style Whitten et al. 2008 / Attreya et al. 2005
                decay = self._particle_alt_decay[specie_idx]
                #mix = self._mie_particle_mix_ratio[idx]*(1-np.exp(decay*(pressure_profile-top_pressure)/(bottom_pressure-top_pressure)))
                mix = self._mie_particle_mix_ratio[specie_idx]*(press/bottom_pressure)**(- decay)
                sigma_xsec_int[cloud_filter, :] = sigma_mie[None] * mix[cloud_filter, None]
            else:
                sigma_xsec_int[cloud_filter, ...] = sigma_mie * self._mie_particle_mix_ratio[specie_idx]

            sigma_xsec += sigma_xsec_int

        self.sigma_xsec = sigma_xsec

        self.debug('final xsec %s', self.sigma_xsec)

        yield 'PyMieScattGridExt', sigma_xsec

    def write(self, output):
        contrib = super().write(output)

        if self._mie_particle_mean_radius is not None:
            contrib.write_array('particle_mean_radius', np.array(self._mie_particle_mean_radius))
        if self._mie_particle_std_radius is not None:
            contrib.write_array('particle_std_radius', np.array(self._mie_particle_std_radius))
        contrib.write_array('particle_mix_ratio', np.array(self._mie_particle_mix_ratio))
        contrib.write_array('particle_midP', np.array(self._mie_midP))
        contrib.write_array('particle_rangeP', np.array(self._mie_rangeP))
        contrib.write_string_array('cloud_species', self._species)
        contrib.write_scalar('radius_Nsampling', self._Nsampling)
        contrib.write_scalar('radius_Dsampling', self._Dsampling)
        contrib.write_scalar('mie_nMedium', self._mie_nMedium)
        return contrib

    @classmethod
    def input_keywords(self):
        return ['PyMieScattGridExtinction', ]
    
    BIBTEX_ENTRIES = [
        """
        @BOOK{1983asls.book.....B,
               author = {{Bohren}, Craig F. and {Huffman}, Donald R.},
                title = "{Absorption and scattering of light by small particles}",
                 year = 1983,
               adsurl = {https://ui.adsabs.harvard.edu/abs/1983asls.book.....B},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        @ARTICLE{2018JQSRT.205..127S,
               author = {{Sumlin}, Benjamin J. and {Heinson}, William R. and {Chakrabarty}, Rajan K.},
                title = "{Retrieving the aerosol complex refractive index using PyMieScatt: A Mie computational package with visualization capabilities}",
              journal = {\jqsrt},
             keywords = {Aerosol optics, Mie theory, Python 3, Electromagnetic scattering and absorption, Open-source software, Physics - Optics, 78-04},
                 year = 2018,
                month = jan,
               volume = {205},
                pages = {127-134},
                  doi = {10.1016/j.jqsrt.2017.10.012},
        archivePrefix = {arXiv},
               eprint = {1710.05288},
         primaryClass = {physics.optics},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2018JQSRT.205..127S},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """,
    ]
