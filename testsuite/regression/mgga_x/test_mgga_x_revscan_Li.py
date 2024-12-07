
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037412009954813e+00, -1.413075232998931e+00, -3.263052519101050e-01, -1.840574996811007e-01, -7.184056216250827e-02, -5.896326707485014e-03, -2.161265237048385e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.676833183084058e+00, -2.679312704007411e+00, -1.836471852647303e+00, -1.838137455061450e+00, -2.762447275957589e-01, -3.237516593936379e-01, -2.434452389310129e-01, 1.716881700126048e+00, -8.385254160592706e-02, 6.641638272024438e+00, -9.376189851813885e-03, 1.705256741453706e+00, -4.621245054334650e-05, 8.118019501857113e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.880648698339818e-04, 0.000000000000000e+00, -1.873839664878700e-04, -7.349464997047652e-04, 0.000000000000000e+00, -7.316878984526870e-04, -2.330374721216357e-01, 0.000000000000000e+00, -1.680577946427543e-01, -3.009337439096547e+00, 0.000000000000000e+00, -4.423309146111157e+04, -8.166230065862420e+01, 0.000000000000000e+00, -1.346963516150824e+10, 2.012141858216744e+01, 0.000000000000000e+00, -3.763648385001434e+04, 3.221764625296219e+04, 0.000000000000000e+00, -1.569643359332462e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscan_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.013475141900393e-02, 1.012535009190450e-02, 1.355609678702012e-02, 1.353314436337177e-02, 5.896450583727592e-02, 4.336268757876297e-02, 1.178092720074509e-01, 5.652900006461418e-01, 2.125389768749577e-01, 5.488017199821031e+00, 1.944576442645467e-11, 5.472047635782255e-01, 2.341987213306889e-23, 6.847289318633865e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
