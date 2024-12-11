
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_18_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.993514778332794e+00, -1.310103619130504e+00, -2.421572605989154e-01, -1.839450410540251e-01, -5.279934648905805e-02, -9.779817031248724e-03, -1.828327213831853e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_18_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.836318508906563e+00, -2.839062737017586e+00, -1.978142548115595e+00, -1.979858976635136e+00, -3.201227079986572e-01, -3.203004663949922e-01, -2.552982432245108e-01, -1.190902798788983e-02, -7.542071525926099e-02, -3.776212794995970e-04, -1.252061361456051e-02, -1.243222993787274e-02, -2.522291892990530e-04, -1.885665999609974e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.848020392291539e-04, 0.000000000000000e+00, -7.821581007079891e-04, -2.853673951874729e-03, 0.000000000000000e+00, -2.846486711271128e-03, -2.725318213094197e-02, 0.000000000000000e+00, -2.901419230477940e-02, -1.241943259016434e+01, 0.000000000000000e+00, -1.382019541211898e+01, -6.258052823576799e+01, 0.000000000000000e+00, -3.456522516115017e+04, -2.566402442243461e-01, 0.000000000000000e+00, -1.235797432305567e+01, -5.234143551205039e-01, 0.000000000000000e+00, 2.042419786490744e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.561632622479461e-02, 2.558609704315270e-02, 3.396220089537658e-02, 3.395503519818345e-02, -5.174271188398933e-04, -4.622506391002739e-04, 2.833769568153353e-01, 1.770170856524248e-04, 4.204548835300757e-02, 1.408320055428617e-05, 3.811660553547743e-06, 1.801053362153173e-04, 6.355090578489358e-11, -6.081884953856840e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
