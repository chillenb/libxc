
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_spbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_spbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.206519274406257e-02, -1.950026604547669e-02, -1.073968610664441e-02, -1.162177339240582e-03, -5.897357339414936e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_spbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_spbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.697192896243956e-02, 1.555922948130705e+00, -3.881805345248900e-02, 4.582091433760600e+01, -2.691905955419129e-02, 2.340039504307487e+01, -4.054282434478808e-03, 8.566072249339391e-01, -2.290130357166095e-06, 1.098545893467458e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_spbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_spbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.631987650324881e-02, 3.263975300649764e-02, 1.631987650324881e-02, 8.401039349228326e-03, 1.680207869845665e-02, 8.401039349228326e-03, 3.407263101661765e-02, 6.814526203323530e-02, 3.407263101661765e-02, 3.167691156276067e-01, 6.335382312552138e-01, 3.167691156276067e-01, 1.256195488648448e+00, 2.512390977297374e+00, 1.256195488648448e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
