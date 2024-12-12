
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r4scan_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.297851415227550e-01, -6.540567691485700e-01, -3.815552432092350e-01, -9.699644107768744e-02, -2.170894852980513e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r4scan_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.726832955453403e-01, -8.377901829304996e-17, -8.565426786212760e-01, -1.163894490615386e-16, -4.859879910789581e-01, -1.627573169129993e-17, -6.118599492936067e-02, -2.220668949757725e-17, 1.151189865268888e-01, 1.966739424398789e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r4scan_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.487093868619096e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.043811953297281e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.490238172301550e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.239435760845215e+01, 0.000000000000000e+00, 0.000000000000000e+00, -2.528800825789063e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r4scan_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.546721748372759e-02, 0.000000000000000e+00, 3.637518633863464e-02, 0.000000000000000e+00, 5.467689952433184e-02, 0.000000000000000e+00, 1.543658989197038e-01, 0.000000000000000e+00, 1.738259713943428e-01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
