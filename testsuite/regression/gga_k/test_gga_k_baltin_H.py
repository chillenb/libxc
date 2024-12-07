
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_baltin_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_baltin", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.039701275796912e+00, 1.896164832374407e+00, 8.367280268785265e-01, 3.223713541678752e-01, 3.806432770744237e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_baltin_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_baltin", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.384315775073926e+00, 4.375206422428021e-16, 2.464606387381296e+00, 4.609845135815782e-16, 6.551064198930090e-01, 1.541909147960439e-16, -2.159742403286244e-01, 1.691235771437822e-16, -3.804071436976615e-01, -7.327669972452411e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_baltin_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_baltin", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.329388435339895e-01, 0.000000000000000e+00, 0.000000000000000e+00, 3.231309295976977e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.614888485152506e+00, 0.000000000000000e+00, 0.000000000000000e+00, 8.478689149213260e+01, 0.000000000000000e+00, 0.000000000000000e+00, 8.109146667132962e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
