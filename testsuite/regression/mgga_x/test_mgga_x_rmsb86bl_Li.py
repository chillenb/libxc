
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rmsb86bl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.905352631572976e+00, -1.342261489519060e+00, -4.032452043841573e-01, -1.711367563349716e-01, -7.847771831744159e-02, -4.193712059716759e-02, -2.592067620134277e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rmsb86bl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.464662763679756e+00, -2.466975985512175e+00, -1.687744410209178e+00, -1.689439244453289e+00, -2.706981893071707e-01, -3.680507565550148e-01, -2.241702194859318e-01, -3.808704741162947e-02, -8.007562461431428e-02, -2.949692161557883e-03, -3.948513036865017e-02, -3.945567857480805e-02, -2.324628231899838e-03, -1.804423911911614e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsb86bl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.280820562994249e-04, 0.000000000000000e+00, -1.274970965290428e-04, -5.350006548729810e-04, 0.000000000000000e+00, -5.303249286676118e-04, -2.104657710814125e-01, 0.000000000000000e+00, -8.261582475096269e-02, -2.057723659880990e+00, 0.000000000000000e+00, -1.531959616140053e+02, -5.374612635758568e+01, 0.000000000000000e+00, -1.262392628505505e+06, -1.368383004951852e+02, 0.000000000000000e+00, -1.353797428808281e+02, -2.928593341034702e+06, 0.000000000000000e+00, -7.666703047970830e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsb86bl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.889761904937229e-06, 6.678333457989082e-12, 5.052924530921757e-05, 1.008964802574469e-17, 3.050742439301710e-02, 9.474173849572982e-11, 4.519900671854787e-03, 1.331487878863168e-16, 6.647331675301991e-07, 5.084816222724171e-11, 4.508316359962000e-17, 4.877779946328005e-17, 3.753812954446780e-31, 6.420763410066483e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
