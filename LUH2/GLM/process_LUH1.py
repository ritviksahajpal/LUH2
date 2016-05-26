import logging
import os
import pdb

import numpy
import pandas

import constants
import pygeoutil.util as util


def add_to_list(arr):
    new_arr = []
    for yr in range(850, 1500):
        new_arr.extend([numpy.nan])

    new_arr.extend(arr)

    for yr in range(2005, 2015):
        new_arr.extend([numpy.nan])

    return new_arr


def diag_LUH1():
    carea = util.open_or_die(constants.CELL_AREA_H)
    halfdeg = util.open_or_die(constants.input_dir + os.sep + '/public_inputs/other/miami_biomass_v3/miami_halfdeg_conform.txt')

    vba = halfdeg * 0.75  # Get above ground biomass
    vba = vba*(vba > 0.01) + (vba<0.01)*0.01  # Set least value of vba to 0.01

    icew = util.open_or_die(constants.input_dir + os.sep + '/gicew.1700.txt', skiprows=6)

    cum_net_C_focal = []
    gross_trans_focal = []
    net_trans_focal = []
    sec_area_focal = []
    sec_age_focal = []
    wh_focal = []

    for yr in range(1500, 2005):
        secd = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gsecd.' + str(yr) + '.txt', skiprows=6)
        ssmb = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gssmb.' + str(yr) + '.txt', skiprows=6)
        ssma = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gssma.' + str(yr) + '.txt', skiprows=6)
        othr = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gothr.' + str(yr) + '.txt', skiprows=6)
        crop = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gcrop.' + str(yr) + '.txt', skiprows=6)
        past = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gpast.' + str(yr) + '.txt', skiprows=6)
        flcp = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflcp.' + str(yr) + '.txt', skiprows=6)
        flpc = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflpc.' + str(yr) + '.txt', skiprows=6)
        flsp = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflsp.' + str(yr) + '.txt', skiprows=6)
        flps = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflps.' + str(yr) + '.txt', skiprows=6)
        flsc = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflsc.' + str(yr) + '.txt', skiprows=6)
        flcs = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflcs.' + str(yr) + '.txt', skiprows=6)
        flvc = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflvc.' + str(yr) + '.txt', skiprows=6)
        flvp = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gflvp.' + str(yr) + '.txt', skiprows=6)
        fvh1 = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gfvh1.' + str(yr) + '.txt', skiprows=6)
        fvh2 = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gfvh2.' + str(yr) + '.txt', skiprows=6)
        fsh1 = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gfsh1.' + str(yr) + '.txt', skiprows=6)
        fsh2 = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gfsh2.' + str(yr) + '.txt', skiprows=6)
        fsh3 = util.open_or_die(constants.input_dir + os.sep + '/LUH/LUHa_u2.v1/updated_states/gfsh3.' + str(yr) + '.txt', skiprows=6)

        cum_net_C_focal.append(numpy.ma.sum(numpy.ma.sum(vba*(1-icew)*carea))*1e6*1e3/1e15 - numpy.ma.sum(numpy.ma.sum(secd*carea*ssmb+othr*carea*vba))*1e6*1e3/1e15)
        gross_trans_focal.append(numpy.ma.sum(numpy.ma.sum((abs(flcp)+abs(flpc)+abs(flsp)+abs(flps)+abs(flsc)+
                                           abs(flcs)+abs(flvc)+abs(flvp)+abs(fvh1)+abs(fvh2)+abs(fsh1)+abs(fsh2)+abs(fsh3))*carea)))
        net_trans_focal.append(numpy.ma.sum(numpy.ma.sum(((flsp + flsc + flvc + flvp) - (flps + flcs) + fvh1 + fvh2)*carea)))
        sec_area_focal.append(numpy.ma.sum(numpy.ma.sum(secd*carea)))
        sec_age_focal.append(numpy.ma.sum(numpy.ma.sum(secd*ssma*carea))/sum(sum((secd+1e-12)*carea)))
        wh_focal.append(numpy.ma.sum(abs(fvh1)+abs(fvh2)+abs(fsh1)+abs(fsh2)+abs(fsh3))*carea)

    cum_net_C_focal = add_to_list(cum_net_C_focal)
    gross_trans_focal = add_to_list(gross_trans_focal)
    net_trans_focal = add_to_list(net_trans_focal)
    sec_area_focal = add_to_list(sec_area_focal)
    sec_age_focal = add_to_list(sec_age_focal)
    wh_focal = add_to_list(wh_focal)

    return cum_net_C_focal, gross_trans_focal, net_trans_focal, sec_area_focal, sec_age_focal, wh_focal

if __name__ == '__main__':
    pass
