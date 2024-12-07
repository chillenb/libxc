
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_meyer_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035145421446566e+00, 1.687597903138880e+00, 6.153398405453746e-01, 1.037757607744034e-01, 6.822515128090932e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_meyer_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388871493349058e+00, 1.299922530988936e-16, 2.672816069039281e+00, 2.589707075841322e-16, 8.752998546897687e-01, 1.315610938399112e-16, -2.013157449326135e-02, 5.058774694821980e-17, -6.895818956132741e-01, -4.407875297140769e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_meyer_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.659194304684756e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.495756984896868e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.281721566421693e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.173433376377019e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.459650443707007e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
