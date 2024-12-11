
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_cf22d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.476426153089880e-01, -3.998219299209092e-01, -8.428098158148145e-02, -4.132496327332309e-02, -3.116065507260262e-03, 7.928538049305240e-04, 2.867811819668290e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_cf22d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.544821224043934e+00, 2.554771536401693e+00, 2.868612799227792e-01, 2.909670029829652e-01, -2.140485458626396e-01, -2.155033496181303e-01, -1.263129590103197e-01, 1.049417927822354e-03, -5.114856876469660e-02, -5.723597748911158e-07, 1.341214943259111e-03, 1.129862809845112e-03, -7.268163152524316e-07, 1.470408545461853e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.962918224502147e-03, 0.000000000000000e+00, -1.962455634341169e-03, -2.085917207239118e-03, 0.000000000000000e+00, -2.091617963093612e-03, -1.053138697249627e+00, 0.000000000000000e+00, -1.052219531200466e+00, 6.999351467820489e-01, 0.000000000000000e+00, 1.508685241681361e-01, -6.588446727416833e+02, 0.000000000000000e+00, 1.343307487194396e+00, 6.338887651286996e-05, 0.000000000000000e+00, 1.405015806623695e-01, 6.147136316606731e-11, 0.000000000000000e+00, -2.625669690832277e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.828636275224402e-01, -2.837469309279243e-01, -1.110177287682453e-01, -1.114723472773622e-01, 1.110747824006102e-02, 1.150935262548304e-02, 1.983039832686041e+00, 1.695440705054323e-05, 3.665212984671574e-01, 3.352301803804381e-09, 8.431140057833785e-09, 1.832587899051539e-05, 4.558751305025971e-20, 6.131074931487387e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
