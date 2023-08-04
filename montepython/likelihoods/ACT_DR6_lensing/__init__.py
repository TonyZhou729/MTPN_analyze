import os
import numpy as np
from montepython.likelihood_class import Likelihood

# Likelihood class object for ACT DR6 lensing.
class ACT_DR6_lensing(Likelihood):
    
    # Initialization
    def __init__(self, path, data, command_line):
        
        Likelihood.__init__(self, path, data, command_line)
        act_datapath = os.path.join(self.data_directory, "ACT_DR6_lensing/v1.1/") # Path to act dr6 files        
        
        # UNCORRECTED
        self.binned_clkk = np.loadtxt(act_datapath + "clkk_bandpowers_act.txt") # Convergence powerspec.
        covmat = np.loadtxt(act_datapath + "covmat_act.txt") # Correlation matrix.
        self.inv_covmat = np.linalg.inv(covmat) # Inverse correlation matrix for log likelihood.
        self.binning_mat = np.loadtxt(act_datapath + "binning_matrix_act.txt") # Binning matrix to compare theory clkk to experiment.
        
        # Requisite cosmo arguments
        self.need_cosmo_arguments(data, {"modes": "s"}) # Require scalah mode.
        self.need_cosmo_arguments(data, {"lensing": "yes"}) # Require lensed cls. 
        self.need_cosmo_arguments(data, {"output": "tCl, pCl, lCl"}) # Require lensing potential output.
        self.need_cosmo_arguments(data, {"l_max_scalars": 2999}) # Match shape with binning matrix.

    def loglkl(self, cosmo, data):
        cls = self.get_cl(cosmo, l_max = 2999)
        ell = cls["ell"]
        theory_unbinned_clkk = ((ell*(ell+1))**2)/4.*cls["pp"] # Unbinned convergence powerspec computed by cosmology.        
        theory_binned_clkk = self.binning_mat @ theory_unbinned_clkk # Bin theory convergence powerspec.
        
        # Compute log likelihood and return.
        diff = self.binned_clkk - theory_binned_clkk
        loglkl = -1/2 * np.dot(diff, np.dot(self.inv_covmat, diff))
        return loglkl

        

