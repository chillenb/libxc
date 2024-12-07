
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bcgp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bcgp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220903800796266e-01, -5.800761358613836e-01, -3.632814600454816e-01, -1.362774909790078e-01, -7.396923398165041e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bcgp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bcgp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282080694056915e-01, -9.515545435905900e-17, -7.159668367460179e-01, -2.402984887764923e-16, -3.991118345272556e-01, 4.275246386253883e-17, -1.427366322806621e-01, -6.892490175629687e-17, -9.856455471381226e-03, -3.669247743864210e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bcgp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bcgp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.910849956119256e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.669331195851695e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.862097020757861e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.386088703934643e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.881708485258087e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
