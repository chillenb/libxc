
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_n12_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.883343725746543e+00, -1.334755615519181e+00, -3.929303419444996e-01, -1.543917024838317e-01, -6.979365546232059e-02, -1.339978430693879e-02, -3.813347576241127e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_n12_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.364054218465748e+00, -2.366213176961105e+00, -1.654816521543951e+00, -1.656204173237762e+00, -4.639159027527413e-01, -4.651645886119414e-01, -2.123857369758566e-01, -1.546033626979683e-02, -6.034819933379760e-02, -8.183542867525212e-04, -1.574226488821636e-02, -1.569931653385000e-02, -5.492169666705461e-04, -3.915201824948539e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_n12_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.642671904877719e-04, 0.000000000000000e+00, -2.635834384582067e-04, -6.656767789821839e-04, 0.000000000000000e+00, -6.639315588640424e-04, -4.850922019150151e-02, 0.000000000000000e+00, -4.789301885995795e-02, 2.889945652432979e+00, 0.000000000000000e+00, 5.450179610754510e-01, -7.083930516023008e+01, 0.000000000000000e+00, -2.831393905675234e+00, 6.027908391121204e-01, 0.000000000000000e+00, 5.562850796741158e-01, -2.115121198890484e+00, 0.000000000000000e+00, -3.072555625588163e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
