
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.485033268726061e+00, -1.065339942160139e+00, -3.483112597555117e-01, -1.386754474930954e-01, -6.636134289683764e-02, -1.061596022491548e-02, -2.038846653208242e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.885761489570923e+00, -1.887280175892286e+00, -1.298175666083223e+00, -1.299148243507217e+00, -3.251401186836334e-01, -3.257284249918151e-01, -1.814193572065363e-01, 4.110432303967526e-01, -5.339531753385345e-02, 2.653406189207206e-01, -1.514970828292803e-02, -1.449786602903351e-02, -4.530498627674919e-04, 2.295527943814698e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.296874463100920e-04, 0.000000000000000e+00, -1.292340648493103e-04, -5.700262680403041e-04, 0.000000000000000e+00, -5.680499099120516e-04, -6.905000897792717e-02, 0.000000000000000e+00, -6.875436583242317e-02, -7.365537550524054e-01, 0.000000000000000e+00, 5.840628401951171e+01, -7.665455948815543e+01, 0.000000000000000e+00, 6.963641964591391e+03, -2.055137318890923e-02, 0.000000000000000e+00, 7.152126991389048e-02, -2.165749644676236e+00, 0.000000000000000e+00, 1.361925533603238e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
