import pandas
import os
import pdb
import numpy
import seaborn


class AgNPP:

    def __init__(self, init_prod=0.0, init_resd=0.0, init_agnpp=0.0, init_nppgr=0.0, init_harvfrac=0.0, init_prodfrac=0.0,
                 init_proddecayrate=0.0, init_resddecayrate=0.0, start_yr=0, end_yr=0):
        """
        Constructor
        """
        # Model parameters
        self.InitProduct   = init_prod
        self.InitResd      = init_resd
        self.InitAgNPP     = init_agnpp    # Initial AgNPP
        self.AgNPPGr       = init_nppgr    # Growth rate of AgNPP
        self.HarvFrac      = init_harvfrac
        self.ProdFrac      = init_prodfrac
        self.ProdDecayrate = init_proddecayrate
        self.ResdDecayRate = init_resddecayrate

        # Model values
        self.AgNPP         = 0.0
        self.Harvest       = 0.0
        self.Product       = 0.0
        self.Residue       = 0.0
        self.Sequester     = 0.0
        self.SequesterRate = 0.0
        self.years         = numpy.arange(start_yr, end_yr+1)

    def run_model(self):
        for idx, yr in enumerate(self.years):
            print yr
            self.AgNPP = self.InitAgNPP * self.AgNPPGr
            self.Harvest = self.HarvFrac * self.AgNPP

            if idx == 0:
                self.Product = self.InitProduct + (self.Harvest * self.ProdFrac) - (self.InitProduct * self.ResdDecayRate)
            else:
                self.Product = self.InitProduct + (self.Harvest * self.ProdFrac) - (self.InitProduct * self.ResdDecayRate)



if __name__ == '__main__':
    obj = AgNPP(init_prod=50, init_resd=5, init_agnpp=5, init_nppgr=1.011, init_harvfrac=1, init_prodfrac=0.5,
                 init_proddecayrate=0.9, init_resddecayrate=0.1, start_yr=1, end_yr=150)

    obj.run_model()