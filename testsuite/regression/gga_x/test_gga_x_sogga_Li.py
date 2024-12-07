
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sogga_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.769438598058587e+00, -1.250780953300703e+00, -3.787548684302743e-01, -1.586750795982348e-01, -7.418432382929400e-02, -1.768010207612664e-02, -3.302377215401792e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sogga_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.271716331595379e+00, -2.273833874373166e+00, -1.552087311190448e+00, -1.553462020712069e+00, -3.733888260520473e-01, -3.734736018310523e-01, -2.069754894433291e-01, -2.248679322945141e-02, -7.545724457451632e-02, -7.137523406424432e-04, -2.364420246929724e-02, -2.347324278743045e-02, -4.767461694007113e-04, -3.389230740986326e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sogga_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.482449592583903e-04, 0.000000000000000e+00, -1.477258717774994e-04, -6.055621944538825e-04, 0.000000000000000e+00, -6.035771155878817e-04, -6.383388737744272e-02, 0.000000000000000e+00, -6.370213576500942e-02, -2.264643083941057e+00, 0.000000000000000e+00, -1.163185211039408e-01, -5.132544708940222e+01, 0.000000000000000e+00, -7.444546847791359e-01, -1.181969615988543e-01, 0.000000000000000e+00, -1.103783306233829e-01, -5.419360323975988e-01, 0.000000000000000e+00, -7.757258365523767e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
