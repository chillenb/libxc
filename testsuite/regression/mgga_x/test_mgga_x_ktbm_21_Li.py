
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_21_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.186335349309130e+00, -1.532883145654889e+00, -2.856098215276190e-01, -1.954450079882317e-01, -6.730973656555779e-02, -1.037714696146142e-02, -1.953077630158709e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_21_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.658388845902378e+00, -2.660954347121781e+00, -1.791393660244695e+00, -1.792689581535976e+00, -3.781716762826436e-01, -3.802123722353738e-01, -2.462811997775337e-01, -1.273674563723995e-02, -9.015534636892429e-02, -4.038945698270204e-04, -1.405915585844587e-02, -1.329625623726172e-02, -2.833542160155860e-04, -1.917877242542690e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_21_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.705151076633045e-04, 0.000000000000000e+00, -8.675897728659362e-04, -3.373725407398707e-03, 0.000000000000000e+00, -3.365076642669444e-03, -5.583852851324899e-02, 0.000000000000000e+00, -5.932968744911601e-02, -1.334287634033316e+01, 0.000000000000000e+00, -8.160205741424193e+00, -9.284030271311671e+01, 0.000000000000000e+00, -2.034521864023079e+04, 3.962533158892187e-01, 0.000000000000000e+00, -7.298238267228226e+00, 8.358768047366787e-01, 0.000000000000000e+00, -9.210510706517729e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_21_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_21", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.591802745267086e-02, 3.587783775533062e-02, 5.686831744333080e-02, 5.683826779374553e-02, 2.064600865213378e-02, 2.249542770250054e-02, 3.706147421116694e-01, 1.046642698547707e-04, 3.586651943517514e-01, 8.289452827769632e-06, -1.205159237235157e-07, 1.065196639178763e-04, -8.036262124167280e-16, 4.017928374721774e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
