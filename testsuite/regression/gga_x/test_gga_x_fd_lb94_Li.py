
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_fd_lb94_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_lb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.350445234251672e+00, -2.016658739159155e+00, -2.059149776455720e+00, -1.924311964109842e-01, -2.736981774880389e-01, -6.359421042163612e+00, -9.128037456661206e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_fd_lb94_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_lb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.608489507026780e+00, -1.611023851104307e+00, -7.362268592788133e-01, -7.376666259655578e-01, 9.835437982679071e-01, 9.865832085608137e-01, -1.692497922824586e-01, 2.375012844631140e+00, 9.743121305259771e-02, 1.883234882328743e+00, 2.351242517526782e+00, 2.398369497121589e+00, 2.093402162909531e+00, 1.938915047858523e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_fd_lb94_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_lb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.612761973421572e-03, 0.000000000000000e+00, -2.603887242524504e-03, -1.028418188947664e-02, 0.000000000000000e+00, -1.025098514966819e-02, -1.807389644576906e+00, 0.000000000000000e+00, -1.808677849094370e+00, -4.114955129634292e+01, 0.000000000000000e+00, -1.048503882086747e+05, -1.011752195334195e+03, 0.000000000000000e+00, -9.387110429715233e+09, -8.935384542674778e+04, 0.000000000000000e+00, -9.031395208073199e+04, -3.138824979833417e+10, 0.000000000000000e+00, -9.925293672200909e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
