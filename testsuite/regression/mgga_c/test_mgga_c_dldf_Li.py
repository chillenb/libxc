
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_dldf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.341513591520295e-02, -8.371482262002408e-02, -4.959806172627838e-02, -1.787375410162581e-02, -1.095905018698960e-02, 1.374592332123197e-02, 2.241748772411659e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_dldf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.027364047996975e-01, -1.024940819512639e-01, -9.254539378426756e-02, -9.240668603323737e-02, -5.664537600732748e-02, -5.668890672673783e-02, -2.097037222522163e-02, 4.901002739261655e-01, -1.310472456565099e-02, 2.999566827689890e-01, 1.512776358597513e-02, 1.565140321668401e-02, 1.033816799304232e-04, 6.351169464040680e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.427783586044839e-04, 0.000000000000000e+00, 4.434552385659636e-04, 1.879350437809581e-03, 0.000000000000000e+00, 1.874529199925593e-03, 9.856780767059119e-01, 0.000000000000000e+00, 9.878628372979726e-01, 4.289205401767606e+01, 0.000000000000000e+00, 1.465632538122208e+02, 1.341228764860823e+03, 0.000000000000000e+00, 2.310615717300819e+05, 1.005029013493741e+00, 0.000000000000000e+00, 4.817601067804024e+01, 3.409794664638554e+00, 0.000000000000000e+00, 7.873503522511452e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.518433291258903e-05, -5.539066242810587e-44, -4.690410644091979e-43, -4.675202053706792e-43, -2.213954497501592e-39, -2.333913716078838e-39, -1.610512088773198e-33, -6.956828268041400e-04, -2.182274454357470e-32, -8.971988809102752e-05, -1.489778010176185e-05, -6.999434029559420e-04, -4.140038141884447e-10, -1.866886203342942e-21]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
