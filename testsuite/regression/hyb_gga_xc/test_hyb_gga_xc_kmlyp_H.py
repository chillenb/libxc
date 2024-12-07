
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_kmlyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_kmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.045943262442657e-01, -2.752912548041280e-01, -1.687644390059835e-01, -5.400352493697276e-02, -4.868977631619964e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_kmlyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_kmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.989689613250649e-01, -2.384902913196354e-01, -3.601583386294812e-01, -2.392880488082443e-01, -2.193485563124740e-01, -1.905246072431493e-01, -6.888287846712278e-02, -6.883873265825462e-02, -6.162328618384393e-03, -5.939332467087672e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_kmlyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_kmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.365062756992971e-16, 1.310319412565954e-02, 9.823631699392380e-03, -7.169953496307921e-16, 2.081819107464455e-02, 1.559337955083867e-02, -2.462528741259732e-14, 1.786073638534114e-01, 1.339515652686682e-01, 1.539580570006506e-11, 7.628485929470323e+00, 5.721354767036328e+00, 6.921586459188070e-24, 3.206698822913791e-18, 2.405020520145726e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
