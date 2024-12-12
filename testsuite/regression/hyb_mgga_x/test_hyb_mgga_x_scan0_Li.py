
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_scan0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.528058208978590e+00, -1.059803707116795e+00, -2.445712782564905e-01, -1.380343469571087e-01, -5.388042140141500e-02, -4.452868854608263e-03, -1.645262027131119e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_scan0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.004456612000303e+00, -2.006318818130809e+00, -1.373236650023740e+00, -1.374488671548529e+00, -1.956633890686788e-01, -2.324606089358707e-01, -1.824111034062077e-01, 1.415776247620467e+00, -6.176030235195434e-02, 5.455703913394337e+00, -7.140965107196810e-03, 1.406240207651677e+00, -3.519569141712580e-05, 5.633668264841021e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_scan0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.555277827844888e-04, 0.000000000000000e+00, -1.549652969102736e-04, -6.090622200862896e-04, 0.000000000000000e+00, -6.063909995398404e-04, -1.898044674927795e-01, 0.000000000000000e+00, -1.394123383148183e-01, -2.483993200726239e+00, 0.000000000000000e+00, -3.645612235993509e+04, -6.783537138103983e+01, 0.000000000000000e+00, -1.106447603398384e+10, 1.532459924421888e+01, 0.000000000000000e+00, -3.101941507671846e+04, 2.453716092466888e+04, 0.000000000000000e+00, -1.089286970783787e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_scan0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.352336973768162e-03, 8.344656315379954e-03, 1.117157933134522e-02, 1.115313915613959e-02, 4.783457777363029e-02, 3.573671245541070e-02, 9.704487185869451e-02, 4.658753259031679e-01, 1.751608217484314e-01, 4.508067487717744e+00, 1.508617505306843e-11, 4.509706474135126e-01, 1.816932728571486e-23, 4.751820424833101e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
