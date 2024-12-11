
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lkt_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lkt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.816167113105010e+01, 1.005452046266301e+01, 3.273601610002873e+00, 1.411578968199828e-01, 7.768816590468568e-02, 3.084472388730704e+00, 1.356897369842305e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lkt_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lkt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.360346197241400e+01, 2.365227103371506e+01, 9.487251353431081e+00, 9.508221109574187e+00, -2.905959829122686e+00, -2.916960727885488e+00, 2.040143748738243e-01, -3.057065625943076e+00, -4.711126489874457e-02, -1.210227402337044e+00, -3.032148342681577e+00, -3.137954916182148e+00, -1.418447087986350e+00, -1.185586587718588e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lkt_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lkt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.138256324627149e-02, 0.000000000000000e+00, 1.134893036570780e-02, 3.824101965668966e-02, 0.000000000000000e+00, 3.813504956327414e-02, 4.052350540495525e+00, 0.000000000000000e+00, 4.058229975886917e+00, 1.441833533286012e+01, 0.000000000000000e+00, 7.829811571772351e+04, 3.864227033488351e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072798e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
