
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.377571563289435e+00, -9.863998606614499e-01, -3.154289344787820e-01, -1.283039351063949e-01, -5.993582619123024e-02, -2.453753083449459e-02, -4.785957463002168e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.763681264954214e+00, -1.764997297957758e+00, -1.210215466094027e+00, -1.211081147086008e+00, -3.098275984642114e-01, -3.092760199933582e-01, -1.629730840002818e-01, 8.586498564225959e-01, -6.150006101873656e-02, 5.654863581850860e-01, -3.431123837620161e-02, -3.296991831062945e-02, -1.028937332412166e-03, 4.441423622045745e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.244716231727924e-05, 0.000000000000000e+00, -9.222389984461569e-05, -4.783189190889164e-04, 0.000000000000000e+00, -4.767863342074631e-04, -5.340806518308471e-02, 0.000000000000000e+00, -5.370570879763033e-02, -3.119732283549739e+00, 0.000000000000000e+00, 2.295606908484423e+02, -3.998860652627742e+01, 0.000000000000000e+00, 2.793747512844170e+04, -4.113073161766634e+00, 0.000000000000000e+00, -3.479337379779349e+00, -2.738740767851081e+01, 0.000000000000000e+00, 2.793259005975671e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
