
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.100637449039710e-02, -5.629325295468333e-02, -6.498879409859544e-02, -2.344946768034806e-03, -1.518603161347337e-02, -2.124566338204139e-04, -2.653189862971742e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.343959813771392e-02, -1.329162710197372e-02, 1.051458023500476e-03, 1.263384416414214e-03, -6.820947134485039e-02, -6.826684027960464e-02, 1.780111983757261e-03, -1.571689224368142e-01, -1.229191849435404e-02, -5.721243928703597e-02, -1.334133293532039e-03, -1.341087611263359e-03, -1.830539915124546e-08, -1.610477432387997e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.102570727489512e-05, 1.420514145497902e-04, 7.102570727489512e-05, 5.576812475520593e-04, 1.115362495104119e-03, 5.576812475520593e-04, 2.090114164218041e-01, 4.180228328436082e-01, 2.090114164218041e-01, 3.517557034078657e+00, 7.035114068157315e+00, 3.517557034078657e+00, 1.320676815081565e+02, 2.641353630163130e+02, 1.320676815081565e+02, 4.287985677363469e-01, 8.575971354726938e-01, 4.287985677363469e-01, 2.333132698207498e-04, 4.666265396414997e-04, 2.333132698207498e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.266285287264332e-03, -4.266285287264332e-03, -9.116363258336245e-03, -9.116363258336242e-03, -1.278855002579240e-03, -1.278855002579239e-03, -1.209296612988662e-01, -1.209296612988394e-01, -4.912667931257687e-02, -4.912667927288711e-02, -1.203309980445291e-05, -1.203309980445291e-05, -6.324714924461569e-15, -6.324714924461792e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
