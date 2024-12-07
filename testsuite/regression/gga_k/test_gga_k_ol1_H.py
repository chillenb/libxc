
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ol1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.036777122338479e+00, 1.697366941987491e+00, 6.208673750970587e-01, 9.800295403582103e-02, 7.628750890192160e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ol1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.389415597756156e+00, 9.454743191389587e-17, 2.676607757216083e+00, 2.821885490955946e-16, 8.789289859276785e-01, 9.334857648155317e-17, 1.054022520422846e-02, 4.502825680962397e-17, -7.593402244792873e-02, -6.400222419114813e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ol1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ol1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.995970014831441e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.075906905370077e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.403659889505122e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.719893932007584e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.622767047919874e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
