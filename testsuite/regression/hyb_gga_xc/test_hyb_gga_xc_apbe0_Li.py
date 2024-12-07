
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_apbe0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.409950308272907e+00, -1.012325285077187e+00, -3.216404939519473e-01, -1.351249106843760e-01, -6.293088261053051e-02, -1.540981556183083e-02, -2.878940454918063e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_apbe0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.788317010545818e+00, -1.789799686590301e+00, -1.229425886272965e+00, -1.230359918072821e+00, -3.252768517469004e-01, -3.254388960102503e-01, -1.780745068634759e-01, -1.122483606714070e-01, -6.373147039324045e-02, 2.848468193820378e-01, -2.059889404575573e-02, -2.045060047837808e-02, -4.156167505288801e-04, -2.954656959069907e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_apbe0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_apbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.750470761722913e-04, 9.631505206920672e-05, -1.742842092458120e-04, -7.193727231241756e-04, 3.041939451362789e-04, -7.166039910335123e-04, -5.231185169730530e-02, 4.952525840231388e-03, -5.215922108552652e-02, 3.736438196216869e-01, 7.701684865214072e+00, 3.674919803822170e+00, -4.349463438400431e+01, 1.857168098684350e+01, 8.160960651033378e+00, -1.786631100764512e-01, 2.393696701107906e-04, -1.668284849512339e-01, -8.188694095171545e-01, 2.290694101076713e-06, -1.172127631578008e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
