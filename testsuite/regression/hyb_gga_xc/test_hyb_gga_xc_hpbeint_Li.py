
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hpbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.540048214888214e+00, -1.092795019742982e+00, -3.382503425164716e-01, -1.480080769130422e-01, -6.561446882041283e-02, -1.712041039650982e-02, -3.198822482242961e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hpbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.007527914456857e+00, -2.009166686860704e+00, -1.391220375449086e+00, -1.392271706454592e+00, -3.196306922837843e-01, -3.197702835837323e-01, -1.960853144889285e-01, -1.254267942297524e-01, -6.795747421063075e-02, 4.275209508792813e-01, -2.288105644777046e-02, -2.271665059738922e-02, -4.617962738782761e-04, -3.282951679848468e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hpbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.747119078588019e-05, 8.450900457717979e-05, -8.700645010581085e-05, -4.104348554287118e-04, 2.837680579169650e-04, -4.085759601600274e-04, -6.722034587533503e-02, 8.520042829067664e-03, -6.707987259071627e-02, 8.174432076668594e-01, 5.522368688952657e+00, 2.529405886786702e+00, -4.161711047427262e+01, 2.885607768192028e+01, 1.294765593547918e+01, -2.353005413372618e-01, 5.524995448211986e-04, -2.196934101434658e-01, -1.077658974900309e+00, 5.290252715233887e-06, -1.542558200169691e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
