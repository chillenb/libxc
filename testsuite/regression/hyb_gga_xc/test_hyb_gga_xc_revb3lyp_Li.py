
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_revb3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_revb3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.498351829806636e+00, -1.079967764949528e+00, -3.261959884387021e-01, -1.335741381465422e-01, -6.643662852223428e-02, -9.471412200316900e-02, -3.606510865066376e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_revb3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_revb3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.876873635543378e+00, -1.878391265099181e+00, -1.303087618010836e+00, -1.304007751080807e+00, -3.776018548043526e-01, -3.778608264685366e-01, -1.708356582098702e-01, -1.150197430253941e-01, -6.437634607684256e-02, -4.319100596299727e-02, -3.151619541714276e-02, -3.170845732233345e-02, -5.179287842889979e-03, -4.536623329252232e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_revb3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_revb3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.780137243237312e-04, 4.387164954355437e-06, -1.775108095154001e-04, -6.563955579777549e-04, 3.063431102968813e-05, -6.547832404107181e-04, -4.776441747467573e-02, 4.009960805412396e-02, -4.760875559972759e-02, -2.954652576216519e+00, 3.860753206340553e+00, -8.946283851902257e+02, -5.131876891885313e+01, 1.979829376836915e+01, -3.249701727649914e+07, -7.804264988092546e+02, 6.666321750293232e-02, -7.816717710772083e+02, -9.648014985023277e+07, 0.000000000000000e+00, -2.874039519964811e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
