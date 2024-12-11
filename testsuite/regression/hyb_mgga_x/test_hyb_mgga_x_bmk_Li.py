
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_bmk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.384026098294418e-01, -5.139513924593829e-01, -1.447141745195708e-01, -6.939814883289658e-02, -3.267865362838795e-02, -4.026816270858235e-02, -6.213798737981931e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_bmk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.066437032747284e+00, -1.068493843282656e+00, -5.978446609374672e-01, -5.988566414230594e-01, -2.082356486497381e-01, -2.087075527308471e-01, -1.018245904165282e-01, -5.049427547086362e-02, -4.618873441185232e-02, -1.630252094783762e-03, -5.403377465900800e-02, -5.265371761361911e-02, -1.088964055877729e-03, -1.035772989732260e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_bmk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.226546659103991e-03, 0.000000000000000e+00, -1.222372915942731e-03, -5.240213215587483e-03, 0.000000000000000e+00, -5.225921754773638e-03, -8.552583077125639e-01, 0.000000000000000e+00, -8.490204255686966e-01, -1.640776121299801e+01, 0.000000000000000e+00, -5.823608507243606e+00, -2.833207675464404e+02, 0.000000000000000e+00, -3.742655147035972e+01, -2.492783174492742e-03, 0.000000000000000e+00, -5.524791879398721e+00, -1.708306741383914e-09, 0.000000000000000e+00, -1.076950706317708e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_bmk_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.182750771362456e-02, 1.190640081317498e-02, -1.275061603700920e-02, -1.267259081855266e-02, 2.796481444828818e-03, 2.914731732006084e-03, 2.640089213793980e-01, -1.843058759560192e-09, 2.191364275516347e-02, -1.019542107252907e-15, -2.217585661479591e-14, -2.103771492599221e-09, 6.880852941215867e-25, 1.203907691978329e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
