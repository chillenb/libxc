
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_eb88_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_eb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.221078803393187e-01, -5.791001716020304e-01, -3.599727745406012e-01, -1.459037495488943e-01, -5.946518009661952e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_eb88_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_eb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.281888974755041e-01, -5.515503699170361e-17, -7.197455103923984e-01, -2.332157886242023e-16, -4.052156566191131e-01, -2.266999861923455e-17, -1.030450767335592e-01, -7.518372084581017e-17, -1.511528358139521e-02, -3.958705742254935e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_eb88_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_eb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.976048023906465e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.433372204945549e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.632447608417796e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.029847663898044e+01, 0.000000000000000e+00, 0.000000000000000e+00, -5.127814784777885e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
