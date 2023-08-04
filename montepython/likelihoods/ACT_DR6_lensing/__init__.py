import os
import numpy as np
from montepython.likelihood_class import Likelihood

# Likelihood class object for ACT DR6 lensing.
class ACT_DR6_lensing(Likelihood):
    
    # Initialization
    def __init__(self, path, data, command_line):
        
        Likelihood.__init__(self, path, data, command_line)

        cl_fname = "clkk_bandpowers_act" # Suffixes to be added
        covmat_fname = "covmat_act" # Suffixes to be added
        binmat_fname = "binning_matrix_act.txt" # This one is fine as is.

        if self.cl_option == "": # Default option
            cl_fname += ".txt"
            covmat_fname += "_cmbmarg.txt"
            print("Selected ACT DR6 CMB Lensing likelihoods (default option).")
        else:
            cl_fname += "_{:s}.txt".format(self.cl_option)
            covmat_fname += "_{:s}_cmbmarg.txt".format(self.cl_option)
            print("Selected ACT DR6 CMB Lensing likelihood ({:s} option)".format(self.cl_option))

        print("Will evaluate lensing convergence powerspec in range {:f} < \ell < {:f}.".format(self.lmin, self.lmax))
        # Load files: bandpowers, covariance matrix and binning matrix.
        self.binned_clkk = np.loadtxt(self.data_directory + cl_fname) # Convergence powerspec.
        covmat = np.loadtxt(self.data_directory + covmat_fname) # Covariance matrix.
        self.binmat = np.loadtxt(self.data_directory + binmat_fname) # Binning matrix to compare theory clkk to experiment.
        
        # Find out the lower and upper bin limit based on requested ell limit.
        ell = np.arange(self.binmat[0].size)
        bin_centers = self.binmat @ ell # Bin centers according to binning matrix.
        
        # Array indices corresponding to limits of bins.
        self.bmin = np.where(bin_centers > self.lmin)[0].min()
        self.bmax = np.where(bin_centers < self.lmax)[0].max() + 1

        self.inv_covmat = np.linalg.inv(covmat[self.bmin:self.bmax, self.bmin:self.bmax]) # Inverse covariance matrix for log likelihood.

        # Requisite cosmo arguments
        self.need_cosmo_arguments(data, {"modes": "s"}) # Require scalah mode.
        self.need_cosmo_arguments(data, {"lensing": "yes"}) # Require lensed cls. 
        self.need_cosmo_arguments(data, {"output": "tCl, pCl, lCl"}) # Require lensing potential output.        

    def loglkl(self, cosmo, data):
        cls = self.get_cl(cosmo) # Obtain theoretical cls
        ell = cls["ell"]
        theory_unbinned_clkk = ((ell*(ell+1))**2)/4.*cls["pp"] # Unbinned convergence powerspec computed by cosmology.        

        binmat = np.copy(self.binmat) # May need to modify, copy a local version.
       
        # Match shapes of cl, ell and binning matrix.
        # If computed at ells larger than bin matrix dimension (3000), must truncate cls.
        if ell.size > binmat[0].size:
            ell = ell[:binmat[0].size]
            theory_unbinned_clkk = theory_unbinned_clkk[:binmat[0].size]            
        else: # Computed at ells smaller than bin matrix dimsion, must truncate bin matrix.
            binmat = binmat[:, :ell.size]

        # Now bin the theory cl using the binning matrix.
        theory_binned_clkk = binmat @ theory_unbinned_clkk # Bin theory convergence powerspec.
        
        # Compute log likelihood and return.
        diff = self.binned_clkk[self.bmin:self.bmax] - theory_binned_clkk[self.bmin:self.bmax]
        loglkl = -1/2 * np.dot(diff, np.dot(self.inv_covmat, diff))
        return loglkl

        

