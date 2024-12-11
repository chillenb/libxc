
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_jk_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.215806357822757e-01, -5.962562634688104e-01, -3.637601131905416e-01, -1.158197588551398e-01, -2.275119958539780e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_jk_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.289230329049677e-01, -5.538907135633023e-18, -7.445986366904812e-01, -1.909478528548140e-16, -4.430009716815121e-01, 4.149142450114058e-17, -1.246152970631241e-01, -4.557977682007634e-17, -9.015489137657207e-03, -2.493877355459017e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.380676315752305e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.341480438520906e-02, 0.000000000000000e+00, 0.000000000000000e+00, -9.175253710016081e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.355530184782123e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.703590357837332e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-6.399205357093945e-06, 0.000000000000000e+00, -6.899935664258918e-03, 0.000000000000000e+00, -6.303278213856439e-03, 0.000000000000000e+00, -4.713213284968360e-03, 0.000000000000000e+00, -2.268817436813829e-03, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
