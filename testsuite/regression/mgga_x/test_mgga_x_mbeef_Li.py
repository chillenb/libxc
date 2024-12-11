
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbeef_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.921117061951549e+00, -1.306666384785786e+00, -2.622999540113241e-01, -1.721110154055320e-01, -5.738728661206965e-02, -1.225438819112014e-02, -2.212121547300179e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbeef_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.507393105804710e+00, -2.509433513024868e+00, -1.966727677456343e+00, -1.967886562842717e+00, -3.536826713521923e-01, -3.538195097904569e-01, -2.252749184096638e-01, -1.564257543847743e-02, -8.121304443484041e-02, -4.942491567669733e-04, -1.638838962604917e-02, -1.633328016457445e-02, -3.556305390056492e+02, -1.970654326913015e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.275027232794113e-05, 0.000000000000000e+00, 7.306344439241385e-05, -2.119550580663305e-03, 0.000000000000000e+00, -2.107364792758551e-03, -2.812613148559462e-01, 0.000000000000000e+00, -2.827579358932353e-01, 1.282395573077866e+00, 0.000000000000000e+00, 3.769355421923635e-01, -1.630920932422722e+02, 0.000000000000000e+00, 2.902268243580557e+01, 1.662739914640356e-04, 0.000000000000000e+00, 3.569003824799056e-01, 1.635100777743619e+07, 0.000000000000000e+00, -2.545403485896200e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeef_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeef", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-7.541411630682493e-03, -7.564995831710468e-03, 3.296211538619575e-02, 3.287523015490915e-02, 7.110733323576605e-04, 7.678506950959431e-04, -1.216950624450773e-01, -4.654108142793974e-13, 3.932489854560933e-02, -1.080646569701040e-08, 1.510982936663026e-12, -1.468502539876030e-13, -1.985274848813826e-03, 8.114979490577677e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
