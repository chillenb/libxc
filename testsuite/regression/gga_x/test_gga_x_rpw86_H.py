
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_rpw86_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.218606592332268e-01, -5.803776936174435e-01, -3.711006192723932e-01, -1.388742183314832e-01, -1.969758168715968e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_rpw86_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.284974743990103e-01, -8.315277790612028e-17, -6.978799643601271e-01, -1.587861056992470e-16, -3.908788899284588e-01, 1.905875670128631e-18, -1.345166702857643e-01, -5.910712870918248e-17, -1.583648807907480e-02, -1.196087718299591e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_rpw86_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.971245993081348e-03, 0.000000000000000e+00, 0.000000000000000e+00, -3.528124261379083e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.269586055014946e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.701043923687892e+00, 0.000000000000000e+00, 0.000000000000000e+00, -8.331952330138334e+03, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
