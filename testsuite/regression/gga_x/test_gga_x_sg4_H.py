
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sg4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.221103890672033e-01, -5.795482416122483e-01, -3.590623340047618e-01, -1.321601733928159e-01, -7.398975218576957e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sg4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.281831637688274e-01, -1.101081459964979e-16, -7.200634875789379e-01, -1.991850673600916e-16, -4.128148277407064e-01, -1.358718176244597e-17, -1.099552955586971e-01, -4.696949709827538e-17, -9.864658656389920e-03, -1.002924370743458e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sg4_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.989973531949293e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.446352352291117e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.439975536137553e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.458027672433596e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.128086498734251e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
