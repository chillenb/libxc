
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pw86_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.217901718216854e-01, -5.760617576782532e-01, -3.662704971984817e-01, -1.372078475364612e-01, -1.995783335803571e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pw86_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285939263051959e-01, -6.466583167072988e-17, -7.020406379114308e-01, -2.223063962950696e-16, -3.883609919936109e-01, 5.165441107798378e-17, -1.321014663407949e-01, -5.505120096047750e-17, -1.601898512431819e-02, -2.856333925360420e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pw86_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.050219326110403e-03, 0.000000000000000e+00, 0.000000000000000e+00, -3.067570668701227e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.183926396182721e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.722810341820689e+00, 0.000000000000000e+00, 0.000000000000000e+00, -8.463404843542485e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
