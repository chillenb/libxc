
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_otpss_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.955779700375937e+00, -1.380647603956610e+00, -4.098775230087877e-01, -1.724679672634004e-01, -7.740820477382797e-02, -2.025446091581154e-02, -3.783264591946212e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_otpss_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.450392821488229e+00, -2.452504524173403e+00, -1.727147933637122e+00, -1.728654835279766e+00, -3.588349793504446e-01, -3.552549444856414e-01, -2.399534854315344e-01, -1.306119211836810e-01, -7.895713558229316e-02, -7.249284503877009e-03, -2.708640558042903e-02, -2.689060480404915e-02, -5.461692456692833e-04, -3.882765574663403e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.748332871440636e-04, 9.286454312875124e-05, -5.740423157350176e-04, -1.042642703545227e-03, 1.000395154550578e-04, -1.040556834938832e-03, -7.739051566236901e-02, -1.719967093068052e-02, -8.168159196023239e-02, 1.593107081572007e+01, 6.275976443139328e+01, 3.482840844857104e+02, -4.330332019867163e+01, 2.053536889093085e+01, 4.215192484763696e+04, -1.400542803316148e-01, -1.528155637736005e-04, -1.307153989071057e-01, -6.394800887774378e-01, 1.821433640972754e-06, -9.153486455773479e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.393788315660434e-02, 2.400467599964743e-02, 1.287679510305562e-02, 1.292183504598107e-02, -1.716805007867144e-03, -7.209831314113327e-04, -5.976292672240604e-01, -1.082658747851527e+00, -3.738231989276934e-02, -2.457196333023139e-02, 1.941111320986284e-14, -1.677781095512833e-10, -5.400522052694039e-32, 3.854792730691677e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
