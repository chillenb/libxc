
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.794844335692114e+00, -1.284786785034670e+00, -4.294802992820262e-01, -1.600360684953139e-01, -8.179984655842720e-02, -1.140169316237562e-02, -2.127820881496785e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.240885954987007e+00, -2.243024015732266e+00, -1.514718980851490e+00, -1.516093135321285e+00, -3.865243503057760e-01, -3.867364661557853e-01, -2.052543973939975e-01, -1.457374231382936e-02, -7.368365447798923e-02, -4.598929181543204e-04, -1.540893048609124e-02, -1.525323445524431e-02, -3.071820507310970e-04, -2.183783727950879e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.591474229315261e-04, 0.000000000000000e+00, -2.582478075544839e-04, -1.041328333221889e-03, 0.000000000000000e+00, -1.037953447915667e-03, -9.025075768257536e-02, 0.000000000000000e+00, -9.004890935259036e-02, -3.986659742979942e+00, 0.000000000000000e+00, 6.616090756807678e-01, -7.742586142547515e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.251170006775683e+00, 0.000000000000000e+00, 8.936885332088502e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
