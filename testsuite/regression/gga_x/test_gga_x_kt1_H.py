
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_kt1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.221131169032341e-01, -5.785536636528638e-01, -3.349604197098459e-01, -8.726227791301532e-02, -4.101589095587792e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_kt1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.283951297344117e-01, -1.126140149090695e-16, -7.272756284696048e-01, -2.165673699569037e-16, -4.331127031663195e-01, -1.102495747423006e-18, -1.160831794100329e-01, -3.693591097067323e-17, -5.468747981385866e-03, -1.233551026296600e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_kt1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.005640300156342e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.623184017064704e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.214406033937811e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.995405584225613e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.999999977351767e-02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
