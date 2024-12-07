
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_mpwlyp1w_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mpwlyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.258179795101766e-01, -5.841715461261363e-01, -3.646419753984503e-01, -1.443421032129541e-01, -5.703110817973283e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_mpwlyp1w_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mpwlyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.326181872488674e-01, -2.402526679939283e-01, -7.229815008357384e-01, -2.508643032703871e-01, -4.085864087031209e-01, -1.972147412601594e-01, -1.198771979713068e-01, -4.257198383426508e-02, -1.650589588844149e-03, -2.993989141992457e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_mpwlyp1w_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mpwlyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.512883559822973e-02, 2.573841703254553e-02, 1.929641940952075e-02, -2.561089479820174e-02, 4.089287532519465e-02, 3.062985268914739e-02, -1.682091327286666e-01, 3.508358932834866e-01, 2.631191460634553e-01, -8.142530715315761e+00, 1.498452593288813e+01, 1.123837543524993e+01, 7.196262787583827e+02, 6.298872687866375e-18, 4.724147450286247e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
