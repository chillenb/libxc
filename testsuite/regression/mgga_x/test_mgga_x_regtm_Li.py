
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.853503147548120e+00, -1.210480560597435e+00, -3.509225144681587e-01, -1.715541069091999e-01, -6.637524576604603e-02, -6.558799535747394e-02, -2.689837550171186e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.727641164336536e+00, -2.730242676894346e+00, -1.655265669245467e+00, -1.657480459523882e+00, -3.526693675924518e-01, -3.513475268289894e-01, -2.361770787598366e-01, -2.085620783475747e-02, -7.382307411872319e-02, -2.079140552640161e-03, -4.463124166323680e-02, -2.155148277148766e-02, -1.762524396356608e-02, -3.277364699621977e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.283498529187966e-04, 0.000000000000000e+00, -8.250097375711100e-04, -1.480477984323792e-03, 0.000000000000000e+00, -1.487080104037398e-03, -5.351086437298178e-02, 0.000000000000000e+00, -5.431620537096456e-02, -5.918069836425693e+00, 0.000000000000000e+00, -5.047770576772504e+02, -3.897365163512720e+01, 0.000000000000000e+00, -4.240807331423854e+06, -2.028060325336529e+01, 0.000000000000000e+00, -4.456805425096433e+02, -8.165547660418251e+02, 0.000000000000000e+00, 6.039112092719328e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.671043809827950e-02, 3.666645338306380e-02, 6.143153893655945e-03, 6.258786996539233e-03, -2.115595911382291e-02, -2.148447175456632e-02, 2.099563746361816e-01, 3.895036744553969e-03, -1.228923051826821e-01, 1.039469658727551e-03, 1.812592449838194e-04, 3.912939835595237e-03, 5.963602849930788e-08, -5.528680095231364e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
