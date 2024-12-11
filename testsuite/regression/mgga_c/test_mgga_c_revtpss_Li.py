
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.346251970582074e-02, -8.371482262002407e-02, -4.959806172627841e-02, -1.808613505662302e-02, -1.095911360424506e-02, -1.550193581524719e-11, -8.300292557722951e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026705211809999e-01, -1.025086386694621e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732751e-02, -5.668890672673786e-02, -2.101629755015065e-02, -1.243109554960658e-01, -1.310473963821593e-02, -7.152742107294510e-02, -1.113606823111059e-10, -8.809083556947177e-11, -1.073246087570411e-18, -5.118744759909199e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.359045208956805e-04, 2.718091272326938e-04, 1.359045207739502e-04, 5.803800301148971e-04, 1.160760060229794e-03, 5.803800301148971e-04, 1.624082993170471e-01, 3.248165986340941e-01, 1.624082993170471e-01, 3.449438751010546e+00, 6.898877508101942e+00, 3.449439487957412e+00, 1.223550734328922e+02, 2.447101468657844e+02, 1.223550734328922e+02, 1.306031819256242e-08, 2.843160860021923e-08, 1.421580493552752e-08, 3.696258305292843e-15, 2.024865339056517e-15, 1.479558879703884e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.012374542424982e-09, -1.012374542424981e-09, -3.623645871820187e-77, -3.623645871820187e-77, -1.123345276918218e-69, -1.123345276918216e-69, -3.796317011289467e-10, -3.796317011288629e-10, -2.940008682004267e-25, -2.940008679629012e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
