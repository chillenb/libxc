
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.565506528762366e-02, -2.438854336082034e-02, -1.481762904283233e-02, -2.019636864178317e-04, -3.846315914119575e-08, -1.067638503069930e-03, -5.788917053522284e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.483265355579254e-02, -3.475985308204445e-02, -3.781956381493292e-02, -3.775418074621841e-02, -4.931938850677482e-02, -4.933936153499956e-02, -1.605381494675234e-03, -1.591399194310048e-01, -1.343048660198849e-02, -9.393543880072112e-02, -2.008547655874808e-03, -2.023105915393412e-03, -1.075222065182059e-05, -1.347139219250734e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.688338048429106e-05, 3.376676096858215e-05, 1.688338048429106e-05, 7.432723555084793e-05, 1.486544711016959e-04, 7.432723555084793e-05, 1.843077645214984e-02, 3.686155290429969e-02, 1.843077645214984e-02, 2.124544043810044e+00, 4.249088087620090e+00, 2.124544043810044e+00, 7.837063043306027e+01, 1.567412608661205e+02, 7.837063043306027e+01, 2.991459126025922e+00, 5.982918252051843e+00, 2.991459126025922e+00, 6.472757710678000e+03, 1.294551542135600e+04, 6.472757710678000e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.568366142326898e-03, -1.568366142326898e-03, -2.252129913878023e-03, -2.252129913878023e-03, -7.795966562943723e-03, -7.795966562943719e-03, -8.068160180182379e-02, -8.068160180180597e-02, -1.874213984263762e-01, -1.874213982749572e-01, -1.738969408900188e-10, -1.738969408900188e-10, -1.666241956151309e-19, -1.666241956151310e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
