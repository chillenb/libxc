
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b88_6311g_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88_6311g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.222454407107850e-01, -5.843176072122275e-01, -3.667632021751488e-01, -1.507205151315622e-01, -5.960335506314959e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b88_6311g_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88_6311g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.280089340588364e-01, -5.825609871206560e-17, -7.157715408483311e-01, -1.850267277148178e-16, -4.031228913405684e-01, -2.497044079849929e-17, -1.077711538496102e-01, -7.396421539257809e-17, -1.534818786912408e-02, -3.015404164913053e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b88_6311g_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88_6311g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.533421377564837e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.941085252860561e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.875883451501072e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.048940949853556e+01, 0.000000000000000e+00, 0.000000000000000e+00, -5.123925591373650e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
