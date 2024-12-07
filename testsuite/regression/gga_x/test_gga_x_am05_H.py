
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_am05_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.216227982797160e-01, -5.580503792422019e-01, -3.300153183416630e-01, -1.153109171511270e-01, -3.914198098852808e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_am05_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.288302539479668e-01, -2.403555823013216e-17, -7.399847925074452e-01, -1.974636009543028e-16, -4.208176473622866e-01, 1.575353059518185e-17, -9.427265239541639e-02, -5.680112100647609e-17, -1.287009149334016e-02, -2.358454558794098e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_am05_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.205065402314591e-06, 0.000000000000000e+00, 0.000000000000000e+00, -1.896224280617486e-03, 0.000000000000000e+00, 0.000000000000000e+00, -4.193758977480841e-02, 0.000000000000000e+00, 0.000000000000000e+00, -6.694529925679540e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.141913041434266e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
