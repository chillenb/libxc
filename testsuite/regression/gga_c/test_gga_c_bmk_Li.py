
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_bmk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.581951705370196e-02, -4.672195536731544e-02, -5.840183648538782e-03, -2.482709434258187e-02, 1.216086196705140e-03, 1.103585927895628e-02, 2.004194824321805e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_bmk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.494094523884693e-01, -2.493913658738989e-01, -1.356064740815593e-01, -1.355552662484592e-01, -6.513056277931424e-02, -6.546820097719988e-02, -7.768696036830908e-02, 3.900836603457248e-01, 2.892064091837825e-03, 2.398445532357359e-01, 1.378441498384796e-02, 1.417094517727067e-02, 1.464514613049552e-04, 6.009590363682180e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_bmk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.660261844050494e-04, 0.000000000000000e+00, 2.654048999095832e-04, 4.425628370090796e-04, 0.000000000000000e+00, 4.415122861106903e-04, 2.836706603753528e-02, 0.000000000000000e+00, 2.847304543700072e-02, 2.367420386653949e+01, 0.000000000000000e+00, 9.801066556006703e+01, -3.146515694329186e+00, 0.000000000000000e+00, 1.158692817111227e+04, 1.185224680926002e+00, 0.000000000000000e+00, 1.257418216050483e+00, 1.944275631651126e+00, 0.000000000000000e+00, 3.058105701215200e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
