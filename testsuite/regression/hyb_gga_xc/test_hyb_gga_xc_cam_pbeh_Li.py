
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_pbeh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.581009070133417e-01, -3.368222084892204e-01, -2.438468937771865e-01, -1.321582105365963e-01, -6.498378472349968e-02, -1.643088365115779e-02, -3.070869259409224e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_pbeh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.060674514174011e-01, -4.059678108634491e-01, -3.792429396467618e-01, -3.791802727355821e-01, -2.461986609302406e-01, -2.462838827761191e-01, -1.673454293658182e-01, -1.185249237847013e-01, -6.622075738112991e-02, 3.421548688453032e-01, -2.195630768390093e-02, -2.179867909943071e-02, -4.433243450363254e-04, -3.151633332549746e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_pbeh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.854212620683231e-05, 9.190971700708733e-05, 1.865304239997661e-05, 9.414853338005985e-06, 2.980993506782570e-04, 9.973230511458019e-06, -1.923673419242849e-02, 6.249948659585063e-03, -1.916778324593913e-02, 2.997129225793205e-01, 6.762268918356340e+00, 3.158961614909070e+00, -4.274142040056186e+01, 2.258698854598489e+01, 9.872332128643434e+00, -2.256069630330614e-01, 3.357174600576258e-04, -2.106660237561792e-01, -1.034552039756748e+00, 3.212885779437900e-06, -1.480855498947315e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
