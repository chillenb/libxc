
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fcfo_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.131434278963594e+00, -1.508010768614924e+00, -6.849126931269380e-01, -3.104935622811698e-01, -2.056548626557875e-01, -3.407445933994381e-01, 8.464693939801357e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fcfo_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.930651871155489e+00, -2.925478633652416e+00, -1.956010671196475e+00, -1.955213566725147e+00, -4.288274152379962e-01, -4.286204858910806e-01, -3.621730476196457e-01, -2.262143949730906e-01, -1.734510092744606e-01, -1.681681729977975e-01, -3.585819544234526e-01, -3.573975121427989e-01, -1.300901501911272e+00, -1.960415899889150e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fcfo_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fcfo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.021272106007733e-04, -6.040079339808969e-04, 5.006828325141809e-04, 1.566106287587465e-04, -2.821714618759422e-04, 1.526565779888427e-04, -1.585352663414479e-01, -7.972231819884008e-02, -1.585257684129661e-01, -1.283001224576516e+01, -2.614097472599958e+00, -2.753141228053180e+03, -1.738161682587350e+02, -3.720736307486766e+01, -2.396733301415681e+08, -3.835659367065999e+03, 6.458785431533986e+03, -3.858963415808531e+03, -2.899950803018934e+08, 2.695301797171885e+10, -2.613355296279888e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
