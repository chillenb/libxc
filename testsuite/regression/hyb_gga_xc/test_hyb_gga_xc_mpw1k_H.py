
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1k_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.877586966241269e-01, -3.510345613841117e-01, -2.158566465420063e-01, -8.183547983984318e-02, -2.191164976534317e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1k_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.107567191731454e-01, 1.690083436176813e+00, -4.513977213201711e-01, 7.673802215393304e+01, -2.601467465518941e-01, 4.810944149310160e+01, -6.857602136704910e-02, 4.491607433521927e-01, -8.072828804328291e-04, 1.109794380001380e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1k_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [9.095981356639582e-03, 3.549935063765323e-02, 1.774967531882661e-02, -5.402747106535301e-03, 1.849336943607058e-02, 9.246684718035291e-03, -5.502941960273761e-02, 8.237240863606407e-02, 4.118620431803204e-02, -4.556972383212417e+00, 2.011103719309944e-01, 1.005551859654972e-01, 4.116272377979914e+02, 2.012696392828865e-03, 1.006348196414432e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
