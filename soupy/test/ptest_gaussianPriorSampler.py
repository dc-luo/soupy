# Copyright (c) 2023, The University of Texas at Austin 
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import unittest
import dolfin as dl
import numpy as np 
import matplotlib.pyplot as plt 

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')

from soupy import STATE, PARAMETER, CONTROL, GaussianPriorSampler, MultipleSamePartitioningPDEsCollective

from mpi4py import MPI



class TestGaussianPriorSampler(unittest.TestCase):
    def setUp(self):
        self.a_tol = 1e-6
        self.nx = 20
        self.ny = 20
        self.gamma = 0.1
        self.delta = 1.0 
        
        # Make spaces
        self.comm_mesh = MPI.COMM_SELF
        self.comm_sampler = MPI.COMM_WORLD

        self.mesh = dl.UnitSquareMesh(self.comm_mesh, self.nx, self.ny)
        self.Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)


    def testSample(self):
        """
        Check expected behavior of sampling - should match 
        usual approach of sampling from hippylib priors
        """
        SEED = 111 
        N_SAMPLES = 5 

        prior = hp.BiLaplacianPrior(self.Vh_PARAMETER, self.gamma, self.delta)
        sampler = GaussianPriorSampler(prior, SEED)

        prior_test = hp.BiLaplacianPrior(self.Vh_PARAMETER, self.gamma, self.delta)
        noise = dl.Vector(self.comm_mesh)
        prior_test.init_vector(noise, "noise")

        m = dl.Function(self.Vh_PARAMETER).vector()
        m_test = dl.Function(self.Vh_PARAMETER).vector()
    
        rng_test = hp.Random(seed=SEED)
        for i in range(N_SAMPLES):
            # Sample from the prior 
            rng_test.normal(1.0, noise)
            prior_test.sample(noise, m_test)

            # Sample using the sampler
            sampler.sample(m)

            # Compare the two 
            diff = m_test.get_local() - m.get_local()
            err = np.linalg.norm(diff)

            print("Sample %d: Error between sampling procedures: %g" %(i, err))
            self.assertTrue(err < self.a_tol)

    def testBurn(self):
        """
        Check that burning indeed burns the sampler 
        by checking that sampling after burning :code:`N_BURN` 
        samples is equivalent to finding the :code:`N_BURN+1` th sample
        obtained by sampling the prior 
        """
        SEED = 111 
        N_BURN = 5 

        prior = hp.BiLaplacianPrior(self.Vh_PARAMETER, self.gamma, self.delta)
        sampler = GaussianPriorSampler(prior, SEED)

        prior_test = hp.BiLaplacianPrior(self.Vh_PARAMETER, self.gamma, self.delta)
        noise = dl.Vector(self.comm_mesh)
        prior_test.init_vector(noise, "noise")

        m = dl.Function(self.Vh_PARAMETER).vector()
        m_test = dl.Function(self.Vh_PARAMETER).vector()

        # Burn from the sampler
        for i in range(N_BURN):
            sampler.burn()
    
        # Burn from the prior 
        rng_test = hp.Random(seed=SEED)
        for i in range(N_BURN):
            rng_test.normal(1.0, noise)
            prior_test.sample(noise, m_test)

        # Check next sample matches
        sampler.sample(m)
        rng_test.normal(1.0, noise)
        prior_test.sample(noise, m_test)

        diff = m_test.get_local() - m.get_local()
        err = np.linalg.norm(diff)
        print("Error between burning procedures: %g" %(err))
        self.assertTrue(err < self.a_tol)


    def testParallelSample(self):
        """
        Check that parallel sampling starting at the same seed is the same 
        """

        SEED = 111 
        N_SAMPLES = 5 

        prior = hp.BiLaplacianPrior(self.Vh_PARAMETER, self.gamma, self.delta)
        sampler = GaussianPriorSampler(prior, SEED)

        m = dl.Function(self.Vh_PARAMETER).vector()
        received_m_np = np.zeros(m.get_local().shape[0] * self.comm_sampler.Get_size())

        for i in range(N_SAMPLES):
            sampler.sample(m)
            self.comm_sampler.Allgather(m.get_local(), received_m_np)
            received_m_np = np.reshape(received_m_np, (self.comm_sampler.Get_size(), m.get_local().shape[0]))
            for j in range(self.comm_sampler.Get_size()):
                diff = m.get_local() - received_m_np[j]
                err = np.linalg.norm(diff)
                self.assertTrue(err < self.a_tol)


    def testParallelBurn(self):
        """
        Check a sampling procedure in parallel where each sampler burns
        samples according to its rank. And check that this matches the samples
        obtained by repeatedly sampling the sampler on a fixed process
        """

        SEED = 111 
        n_proc = self.comm_sampler.Get_size()
        i_proc = self.comm_sampler.Get_rank()

        prior = hp.BiLaplacianPrior(self.Vh_PARAMETER, self.gamma, self.delta)
        sampler = GaussianPriorSampler(prior, SEED)

        m = dl.Function(self.Vh_PARAMETER).vector()
        received_m_np = np.zeros(m.get_local().shape[0] * self.comm_sampler.Get_size())

        # first burn according to rank
        for i in range(i_proc):
            sampler.burn()

        # sample next one 
        sampler.sample(m)
        self.comm_sampler.Allgather(m.get_local(), received_m_np)
        received_m_np = np.reshape(received_m_np, (n_proc, m.get_local().shape[0]))

        # check against sampling from the beginning
        new_sampler = GaussianPriorSampler(prior, SEED)
        for i in range(n_proc):
            new_sampler.sample(m)
            diff = m.get_local() - received_m_np[i]
            err = np.linalg.norm(diff)
            self.assertTrue(err < self.a_tol)




if __name__ == "__main__":
    unittest.main()
