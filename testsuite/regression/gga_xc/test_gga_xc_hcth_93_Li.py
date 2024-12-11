
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_93_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.916654694548436e+00, -1.343184339038492e+00, -4.708500462100610e-01, -1.729047801630669e-01, -8.833376722553364e-02, -3.224191113814738e-02, -6.537517619775833e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_93_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.567376148414667e+00, -2.569665729609369e+00, -1.743683008420777e+00, -1.745183250533756e+00, -3.146855363143091e-01, -3.150589427018160e-01, -2.322923837711666e-01, 3.951054611595708e-01, -6.060354454452896e-02, 2.728176463252034e-01, -4.354782036623972e-02, -4.270388523472134e-02, -1.105037506244866e-03, -2.184130396219183e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_93_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.403927577500951e-05, 0.000000000000000e+00, 4.420113095502844e-05, -1.752641903053365e-04, 0.000000000000000e+00, -1.737181802978241e-04, -1.506642511551574e-01, 0.000000000000000e+00, -1.504664408784374e-01, 1.044094118487554e+00, 0.000000000000000e+00, 7.459154340455105e+01, -1.224849573920743e+02, 0.000000000000000e+00, 9.155770204644101e+03, -2.230596683177148e+00, 0.000000000000000e+00, -1.964335610631651e+00, -1.306842808707960e+01, 0.000000000000000e+00, 3.306833152920635e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
