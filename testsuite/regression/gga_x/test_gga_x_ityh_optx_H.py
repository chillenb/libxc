
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ityh_optx_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.304919619656298e-01, -1.012144568673680e+00, -4.641735393262453e-01, -2.694039892875597e-02, -3.361527509554914e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ityh_optx_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.611107412094292e-01, -4.287759314714538e-17, -1.426231175161709e+00, -4.480194170864285e-16, -7.095915013633947e-01, -1.092474602311537e-16, -5.094690270612386e-02, -1.329948258399428e-17, -6.722097747702188e-06, -1.221235778916069e-22]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ityh_optx_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.203948178350183e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.902057608948890e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.125288637609325e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.615889592339612e-04, 0.000000000000000e+00, 0.000000000000000e+00, -2.148149213952880e-09, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
