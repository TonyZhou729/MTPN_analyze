#!/usr/bin/python
"""
.. module:: MontePython
   :synopsis: Main module
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Monte Python, a Monte Carlo Markov Chain code (with Class!)
"""
import os
import sys

# Checking for python version, comment if you are tired of seeing it when using
# a version < 2.7
version = sys.version[:3]
if float(version) < 2.7:
    print '\n\n /|\  You must have Python >= 2.7,'
    print '/_o_\ or install manually the following modules for your older'
    print '      distribution: argparse and OrderedDict'

import parser_mp   # parsing the input command line
import io_mp       # all the input/output mechanisms
import mcmc        # the actual Monte Carlo chain procedure
import data        # data handling


def main():
    """
    Main call of the function

    This function recovers the input from the command line arguments, the
    parameter files, and launch accordingly a Markov chain. 
    
    It first extract the path of the used Monte Python code, assuming a
    standard setting (the data folder is in the same directory as the code
    folder).
    """
    # Parsing line argument
    command_line = parser_mp.parse()

    # Default configuration
    path = {}

    # On execution, sys.path contains all the standard locations for the
    # libraries, plus, on the first position (index 0), the directory from
    # where the code is executed. By default, then, the data folder is located
    # in the same root directory. Any setting in the configuration file will
    # overwrite this one.
    path['MontePython'] = sys.path[0] + '/'
    path['data'] = path['MontePython'][:-5] + 'data/'

    # Configuration file, defaulting to default.conf in your root directory.
    # This can be changed with the command line option -conf. All changes will
    # be stored into the log.param of your folder, and hence will be reused for
    # an ulterior run in the same directory
    conf_file = path['MontePython'][:-5] + command_line.config_file
    if os.path.isfile(conf_file):
        for line in open(conf_file):
            exec(line)
        for key, value in path.iteritems():
            if not value.endswith('/'):
                path[key] = value + '/'
    else:
        print ' /|\  You must provide a .conf file (default.conf by default)'
        print '/_o_\ in your montepython directory that specifies the correct'
        print '      locations for your data folder, Class (, Clik), etc...'

    sys.stdout.write('Running MontePython version 1.1\n')

    # If the info flag was used, read a potential chain (or set of chains) to
    # be analysed with default procedure. If the argument is a .info file, then
    # it will extract information from it (plots to compute, chains to analyse,
    # etc...)
    if command_line.files is not None:
        import analyze    # analysis module, only invoked when analyzing
        analyze.info(command_line)
        exit()

    # If the restart flag was used, load the cosmology directly from the
    # log.param file, and append to the existing chain.
    if command_line.restart is not None:
        if command_line.restart[0] == '/':
            folder = ''
        else:
            folder = './'
        for elem in command_line.restart.split("/")[:-1]:
            folder += ''.join(elem+'/')
        command_line.param = folder+'log.param'
        command_line.folder = folder
        sys.stdout.write('Reading {0} file'.format(command_line.restart))
        Data = data.data(command_line, path)

    # Else, fill in data, starting from  parameter file. If output folder
    # already exists, the input parameter file was automatically replaced by
    # the existing log.param. This prevents you to run different things in a
    # same folder.
    else:
        Data = data.data(command_line, path)

    # Overwrite arguments from parameter file with the command line
    if command_line.N is None:
        try:
            command_line.N = Data.N
        except AttributeError:
            print '\n /|\  You did not provide a number of steps,'
            print '/_o_\  neither via command line, neither in ',
            print command_line.param
            exit()

    # Creating the file that will contain the chain
    io_mp.create_output_files(command_line, Data)

    # If there is a conflict between the log.param value and the .conf file,
    # exiting.
    if Data.path != path:
        print ' /|\  Your log.param file is in contradiction with your .conf'
        print '/_o_\ file. Please check your path in these two places.'
        exit()

    # Loading up the cosmological backbone. For the moment, only Class has been
    # wrapped.

    # Importing the python-wrapped Class from the correct folder, defined in
    # the .conf file, or overwritten at this point by the log.param.
    # If the cosmological code is Class, do the following to import all
    # relevant quantities
    if Data.cosmological_module_name == 'Class':
        for elem in os.listdir(Data.path['cosmo']+"python/build"):
            if elem.find("lib.") != -1:
                classy_path = path['cosmo']+"python/build/"+elem

        # Inserting the previously found path into the list of folders to
        # search for python modules.
        sys.path.insert(1, classy_path)
        try:
            from classy import Class
        except ImportError:
            print(" /|\  You must have compiled the classy.pyx")
            print("/_o_\ please go to /path/to/class/python and run")
            print("      python setup.py build")
            exit()

        cosmo = Class()
    else:
        print(" /|\  Error: unrecognised cosmological module")
        print("/_o_\ Be sure to define the correct behaviour")
        print("      in MontePython.py, and data.py")
        exit()

    # MCMC chain
    mcmc.chain(cosmo, Data, command_line)

    # Closing up the file
    Data.out.close()


#-----------------MAIN-CALL---------------------------------------------
if __name__ == '__main__':
    main()
