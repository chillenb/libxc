
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86vwn_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.202367750053537e-02, -1.834130571389092e-02, -6.485504376674011e-03, 7.775377977862687e-03, -1.560270494514471e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86vwn_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.685275827594832e-02, -2.703381530065603e-01, -4.454404332600780e-02, -2.567437286045595e-01, -3.800501433698387e-02, -1.950413047753319e-01, 2.826921857994963e-03, -5.799040699976808e-02, -1.992580292529131e-03, -6.955988126073802e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86vwn_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.508879410873103e-02, 3.017758821746206e-02, 1.508879410873103e-02, 1.196860506134854e-02, 2.393721012269707e-02, 1.196860506134854e-02, 7.183876399931417e-02, 1.436775279986283e-01, 7.183876399931417e-02, 1.061364826031581e+00, 2.122729652063161e+00, 1.061364826031581e+00, -1.065518117576132e+00, -2.131036235152263e+00, -1.065518117576132e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
