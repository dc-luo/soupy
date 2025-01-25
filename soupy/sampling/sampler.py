from abc import ABC, abstractmethod

import dolfin as dl 
import numpy as np 
import hippylib as hp 



class ParameterSampler(ABC):
    """
    Base class for a sampling distribution for the uncertain parameter
    """

    @abstractmethod
    def sample(self, m):
        """
        Draw a sample from the distribution and place into a :code:`dl.Vector` 

        :param m: Destination for sample 
        :type m: :code:`dl.Vector`
        """

    @abstractmethod
    def burn(self):
        """
        Consumes the internal random state to burn one sample 
        """







class GaussianPriorSampler:
    """
    Wrapper for the Gaussian Prior distributions from :code:`hippylib`
    """
    def __init__(self, prior, seed=1):
        """
        Constructor

        :param prior: a Gaussian prior with a sample method 
        :type prior: hp.Prior 

        :param seed: Seed for :code:`hp.Random` random number generator from hippylib
        :type rng: hp.Random
        """

        self.Vh = prior.Vh 
        self.mpi_comm = self.Vh.mesh().mpi_comm()
        self.prior = prior 
        self.seed = seed 
        self.rng = hp.Random(seed=seed)
        self.noise = dl.Vector(self.mpi_comm)
        self.prior.init_vector(self.noise, "noise")

    def sample(self, m):
        """
        Sample and save to vector :code:`m`
        """

        self.rng.normal(1.0, self.noise)
        self.prior.sample(self.noise, m)

    def burn(self):
        """
        Burn a sample from the rng 
        """
        self.rng.normal(1.0, self.noise)





class NumpyArrayOnDiskSampler:
    """
    Loads samples from disk. This is useful for 
    the cases where the parameter distribution is given 
    only in terms of samples (e.g. Bayesian calibration process)
    """
    def __init__(self, Vh, data_directory, sample_name='sample'):
        self.Vh = Vh 
        self.data_directory = data_directory 
        self.sample_name = sample_name
        self._sample_index = 0 

    def sample(self, m):
        """
        Sample from the distribution by loading the ith sample from disk, where 
        the sample index is kept track by an internal counter 
        """
        m_np = np.load("%s/%s_%d.npy" %(self.data_directory, self.sample_name, self._sample_index))
        m.set_local(m_np) 
        m.apply('')
        self._sample_index += 1 
    
    def burn(self):
        """
        Burn a sample from the distribution by incrementing the internal sample index counter 
        """
        self._sample_index += 1 











