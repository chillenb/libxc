
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_ccalda_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.348344939320699e-02, -8.371482262002407e-02, -4.959806172627838e-02, -1.808590353262049e-02, -1.095911360425487e-02, -6.777830339329991e-03, -1.131913014452829e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_ccalda_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026428605860029e-01, -1.024807880919013e-01, -9.254539378426763e-02, -9.240668603323743e-02, -5.664537600732744e-02, -5.668890672673781e-02, -2.101619462605608e-02, -1.243107795716159e-01, -1.310473963818738e-02, -7.152742107535329e-02, -8.373119879865047e-03, -8.768747737189038e-03, -6.481224282914944e-05, -5.936313586313902e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.890790540163771e-10, 7.781581080824050e-10, 3.890790540163771e-10, 9.555140264726584e-10, 1.911028052846326e-09, 9.555140264726584e-10, 1.286507051510631e-08, 2.573014103326685e-08, 1.286507051510631e-08, 2.127804742912212e+01, 4.255609485824424e+01, 2.127804742912212e+01, 6.395458812887258e+01, 1.279091762577452e+02, 6.395458812887258e+01, 3.615700608425459e-04, 7.231401216591608e-04, 3.615700608425459e-04, 1.718112593986912e+00, 3.436225187973827e+00, 1.718112593986912e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.419551906565930e-11, -1.419552893804037e-11, 2.170525696907442e-18, 4.791454922934898e-18, -6.965691502490652e-19, -7.211763342232204e-19, -1.249837027406524e-05, -1.249837027440776e-05, -1.499011789452602e-13, -1.498939677442702e-13, -1.073476152146477e-08, -1.073476152216163e-08, -4.172119120510137e-10, -4.172119120510138e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
