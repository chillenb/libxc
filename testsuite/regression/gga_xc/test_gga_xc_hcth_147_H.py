
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_147_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_147", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.955964347991519e-01, -6.098429811979603e-01, -3.665220820597553e-01, -1.773037763775590e-01, -1.436419764983324e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_147_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_147", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.247900487542136e-01, -1.347103764269252e-01, -8.171263446501571e-01, -1.770352886235747e-01, -4.421004963169914e-01, -1.861917352750491e-01, -1.083277029152893e-01, 3.824549665742088e-02, -1.899327362999798e-02, 4.368473050026817e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_147_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_147", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.800123620182335e-02, 0.000000000000000e+00, -1.147232595462673e+21, 3.333521601363447e-03, 0.000000000000000e+00, -7.919701958482345e+20, -9.343949712851131e-02, 0.000000000000000e+00, -2.750609891585444e+20, -1.415490890174048e+01, 0.000000000000000e+00, 6.664895166162448e+19, -3.823536959578604e+01, 0.000000000000000e+00, 1.670316033203899e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
