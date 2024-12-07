
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_vwn_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.199830183826378e-02, -1.811722049095568e-02, -8.045925202926973e-03, -1.586505471658124e-04, -2.287252191032069e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_vwn_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.690908854326146e-02, 1.589508176092201e+00, -4.106856156609381e-02, 7.377778878856249e+01, -2.749652606547753e-02, 4.123094804148656e+01, -9.421718215889690e-04, 3.382721480253117e-01, -1.489988905710326e-09, 1.050080897667822e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_vwn_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.644027617103919e-02, 3.288055234207837e-02, 1.644027617103919e-02, 1.019984830057715e-02, 2.039969660115429e-02, 1.019984830057715e-02, 4.197224264403420e-02, 8.394448528806837e-02, 4.197224264403420e-02, 8.813668952367522e-02, 1.762733790473505e-01, 8.813668952367522e-02, 9.745898094077085e-04, 1.949179619627777e-03, 9.745898094077085e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
