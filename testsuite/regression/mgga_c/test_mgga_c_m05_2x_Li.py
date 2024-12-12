
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m05_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.029766742838908e-02, -4.625013203376284e-02, 1.896968085431969e-02, -2.153642834061092e-04, 1.194554444908875e-07, 3.928004255711660e-02, 9.792736710356506e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m05_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.526635162392394e-02, -8.510785952219507e-02, -7.980310273285650e-02, -7.963401644979136e-02, -7.338279621644042e-02, -7.199867451493663e-02, -1.233290587837567e-02, 1.120111473678735e+00, 2.220771631889465e-02, 6.752767209296215e-01, 5.985313778863909e-02, 5.669035823405653e-02, 1.159071954882479e-03, 2.097597743053843e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.507699387203762e-04, 0.000000000000000e+00, 1.504752790811617e-04, 4.337688647619159e-04, 0.000000000000000e+00, 4.331998255004981e-04, -1.436861614320544e-02, 0.000000000000000e+00, -1.802520650172970e-02, 1.531724994585234e+01, 0.000000000000000e+00, -1.974265000290983e+02, -1.295882606159095e+02, 0.000000000000000e+00, -1.687297227324879e+06, -4.801270054033243e+00, 0.000000000000000e+00, -3.719344527349518e+02, -1.999944824548908e+01, 0.000000000000000e+00, -8.061579140285966e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-8.235520649533135e-03, -8.242661904840671e-03, -7.841983246758708e-03, -7.863724894215432e-03, 2.286671879899676e-02, 2.415483147948460e-02, -5.621558474507869e-01, 5.413404553452415e-03, 3.099072290297117e-01, 6.983701230624318e-04, 2.369747406308205e-06, 5.446451088939682e-03, 2.551740578602872e-14, 3.516748619323667e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
