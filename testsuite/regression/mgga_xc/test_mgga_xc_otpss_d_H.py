
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_otpss_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.695453076756024e-01, -6.020073426336936e-01, -3.512745705063115e-01, -1.006736354925611e-01, -5.688712867263399e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_otpss_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.025118419876386e-01, -6.386557178391235e-02, -7.831166221471898e-01, -2.506001728069861e-01, -4.662767124040287e-01, -1.951180632676889e-01, -1.322013825746927e-01, -9.051731111310302e-02, -7.493556177264768e-03, -7.071279902173429e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.157856204210120e-01, 5.224551940807422e+00, 2.612275970403711e+00, 8.860962323512859e-03, 6.935008322531555e-02, 3.467504161265778e-02, 1.556831745404383e-01, 5.925599985798949e-01, 2.962799992899475e-01, 3.592664177733192e+01, 1.164967748562692e+02, 5.824838742813460e+01, 7.315780867949727e+06, 2.365073882482243e+07, 1.182536941241122e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.013904928805674e+00, -6.157197689718882e+00, -1.670469850980956e-02, -1.735823258748159e-79, 3.328347155822635e-03, -7.376898574840126e-77, -5.920374672652025e-05, -4.487185716425840e-59, -3.259264328635829e-09, -7.264645212571623e-44]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
