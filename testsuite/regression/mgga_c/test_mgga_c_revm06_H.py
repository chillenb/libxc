
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm06_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.361639306244284e-12, -8.900839601752021e-13, -1.871779515830400e-12, 5.400004291361017e-12, 4.755465090565509e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm06_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.387046318492632e-03, -4.315642113509939e-01, 1.397402177614234e-03, -3.843827055062328e-01, 1.095852823992557e-02, -2.666617292480740e-01, 2.180923858039074e-02, 3.631661195444880e-03, 3.984786897317246e-03, 4.055387321302340e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.763781499313719e-02, 0.000000000000000e+00, -7.728643440962438e+16, -1.730877946672528e-03, 0.000000000000000e+00, -7.709161464558355e+16, -6.382052760010788e-02, 0.000000000000000e+00, -5.518398467911412e+16, -6.546258433191189e+00, 0.000000000000000e+00, 3.042966105317803e+16, -8.490759910395873e+03, 0.000000000000000e+00, 7.069043428359334e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.328646851518725e-01, 2.560297943684142e+06, 2.975879969112957e-03, 2.083879119353964e+06, 2.195560188570946e-02, 1.173231149686869e+06, 4.289354409975478e-02, -1.269923342626685e+03, 5.817001092151698e-03, -1.907292629752832e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
