
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.481249212962609e-02, -4.864024140214231e-02, -4.496995857153345e-03, -1.578711481768095e-02, -3.746686705966969e-03, -1.537287484462690e-08, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.171029124213784e-01, -1.169703772824159e-01, -1.044139408460930e-01, -1.043118635518369e-01, -2.203561893698613e-02, -2.204359191039615e-02, -2.355266256728370e-02, -1.035574763219224e-01, -4.042254639608537e-03, 2.719838298213394e-01, -1.232590903057610e-06, 1.107486662213418e-06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.225450228853952e-05, 8.450900457707903e-05, 4.225450228853952e-05, 1.418840289582866e-04, 2.837680579165732e-04, 1.418840289582866e-04, 4.260021414531472e-03, 8.520042829062938e-03, 4.260021414531472e-03, 2.664014736327776e+00, 5.328029472655551e+00, 2.664014736327776e+00, -2.926574824970755e+00, -5.853149649941504e+00, -2.926574824970755e+00, 2.291163411318837e-04, 4.582326822992918e-04, 2.291163411318837e-04, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
