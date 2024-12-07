
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hse03_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse03", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.428760647037197e+00, -1.032687028398627e+00, -3.416105832749333e-01, -1.555997060079399e-01, -7.826773707379764e-02, -2.053159051458186e-02, -3.838587476957801e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hse03_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse03", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.815632989283754e+00, -1.817135077828087e+00, -1.249525590207546e+00, -1.250458992695991e+00, -3.413361637164293e-01, -3.415596556417348e-01, -1.997804595257967e-01, -1.237538161436014e-01, -7.877569249441915e-02, 3.419889375379563e-01, -2.745355931416626e-02, -2.725486253893230e-02, -5.541560847196045e-04, -3.939544638854929e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hse03_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse03", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.527516436375038e-04, 9.190971700708733e-05, -1.520126061206749e-04, -7.028914707021735e-04, 2.980993506782570e-04, -7.001118766215046e-04, -5.395602002692506e-02, 6.249948659585063e-03, -5.377610199040961e-02, 4.276902801480187e-02, 6.762268918356340e+00, 3.381134459178170e+00, -5.081879818937445e+01, 2.258698854598489e+01, 1.129349427299244e+01, 1.678587300264123e-04, 3.357174600576258e-04, 1.678587300264123e-04, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
