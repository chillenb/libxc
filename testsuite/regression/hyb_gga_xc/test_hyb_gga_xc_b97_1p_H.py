
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_1p_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.541956927139718e-01, -5.094607467404538e-01, -3.076028268742822e-01, -1.458447519723581e-01, -1.083550722927522e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_1p_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.330854383159044e-01, -2.408319510141272e-01, -6.691939642207144e-01, -2.438652789851206e-01, -3.683987434469462e-01, -2.038384229599036e-01, -7.911322756408135e-02, 3.425239846125900e-02, -1.440832920919859e-02, 3.088358066877236e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_1p_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.131643082186717e-02, 0.000000000000000e+00, -3.324713217680748e+20, -3.304476911933795e-03, 0.000000000000000e+00, -2.325373646058263e+20, -8.822215077319584e-02, 0.000000000000000e+00, -7.128381459761518e+19, -1.296084721545001e+01, 0.000000000000000e+00, 6.651158693941560e+19, -2.505740939127659e+01, 0.000000000000000e+00, 8.042132613699827e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
