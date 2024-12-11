
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_baltin_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_baltin", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.820908758581313e+01, 9.925463588417745e+00, 2.227336702341033e+00, 1.413359916852296e-01, 6.018770199864201e-02, 1.714279282002323e+00, 7.538321149995506e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_baltin_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_baltin", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.407664604880841e+01, 2.412432856254655e+01, 1.045828462418431e+01, 1.047876858467713e+01, -1.048897820523855e+00, -1.054848984291932e+00, 2.049532488862576e-01, -1.697332160716898e+00, -5.849050140314115e-03, -6.723475131358647e-01, -1.683379449100451e+00, -1.742177508507752e+00, -7.880256943404730e-01, -6.586589800628720e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_baltin_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_baltin", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.070662901712725e-02, 0.000000000000000e+00, 1.067774020776786e-02, 3.199078776374231e-02, 0.000000000000000e+00, 3.190924044063191e-02, 2.307448186188731e+00, 0.000000000000000e+00, 2.310394569346453e+00, 1.447240274971044e+01, 0.000000000000000e+00, 4.349895317651306e+04, 2.323065448201540e+02, 0.000000000000000e+00, 1.363542626209158e+09, 3.740819308599658e+04, 0.000000000000000e+00, 3.823642980040444e+04, 4.575635063887645e+09, 0.000000000000000e+00, 1.273532430599935e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
