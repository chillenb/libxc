
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mggac_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.993773269564785e+00, -1.316756961346107e+00, -2.712651114111688e-01, -1.824808015584055e-01, -5.847095031186642e-02, -1.337844320753615e-02, -1.993622157921318e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mggac_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.784160392826116e+00, -2.786676185456102e+00, -2.006346543559511e+00, -2.007818792365019e+00, -3.622703383908150e-01, -3.621578360675692e-01, -2.483472039803630e-01, -1.702440202814498e-02, -7.869901907732445e-02, -5.399358206332810e-04, -1.790231731833781e-02, -1.777211126097943e-02, -2.878122332846294e-04, -2.046047877826202e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mggac_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.489280801857444e-04, 0.000000000000000e+00, -3.474624956443908e-04, -2.118600659762368e-03, 0.000000000000000e+00, -2.110479972348118e-03, -3.861559171713837e-04, 0.000000000000000e+00, -4.205939365444065e-04, -3.679037944202899e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.582950322018154e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.534420330153170e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mggac_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.810550576903590e-02, 1.807823715939541e-02, 3.679185318057834e-02, 3.674449336104314e-02, 9.297329681298968e-05, 1.011356679852169e-04, 1.412281018120853e-01, 2.330323742747301e-11, 6.177064026466029e-03, 7.425595060796072e-10, 2.218615062077262e-11, 2.232278728590314e-11, 1.157606750745399e-14, 6.693632057552397e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
