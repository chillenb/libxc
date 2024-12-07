
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hse12s_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse12s", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.140364830505516e+00, -8.328487800604243e-01, -2.928486108905610e-01, -1.455298092258504e-01, -7.714947119345091e-02, -2.052547552052057e-02, -3.838587057590288e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hse12s_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse12s", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.449495555866119e+00, -1.450618364274520e+00, -1.011917350375706e+00, -1.012611403912547e+00, -2.953978830255801e-01, -2.955729804375700e-01, -1.831501659525420e-01, -1.237431678908629e-01, -7.713212262672399e-02, 3.419889378767931e-01, -2.744115243594237e-02, -2.724272846313358e-02, -5.541559837458838e-04, -3.939544276069424e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hse12s_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse12s", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.076503471841430e-04, 9.190971700708733e-05, -1.070793277854874e-04, -5.089545923398506e-04, 2.980993506782570e-04, -5.068062478603089e-04, -4.109500112933860e-02, 6.249948659585063e-03, -4.095519231192943e-02, 2.379596364948187e-01, 6.762268918356340e+00, 3.381134459178170e+00, -4.931020913657481e+01, 2.258698854598489e+01, 1.129349427299244e+01, 1.678587300264123e-04, 3.357174600576258e-04, 1.678587300264123e-04, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
