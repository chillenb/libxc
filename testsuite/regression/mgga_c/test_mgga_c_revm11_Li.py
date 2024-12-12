
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.880460126837306e-01, -1.007319724211556e-01, -9.404607163342498e-03, -1.347519494390585e-02, -2.530572791445591e-03, -1.428940251535888e-02, -3.545721161651436e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.788834094607155e-01, -1.786991966403377e-01, -2.851643561179781e-01, -2.850179615911486e-01, -6.237955891406354e-03, -6.254036534528345e-03, 2.839608252446904e-02, -1.312552002105422e-01, -8.303683856108690e-03, 7.329575073488800e-01, -1.795977952943539e-02, -1.816132963436362e-02, -4.171176054789353e-04, -6.120616475590938e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.962808682152138e-04, -5.925617364304276e-04, -2.962808682152138e-04, -1.359815715846657e-04, -2.719631431693312e-04, -1.359815715846657e-04, 8.274152127513945e-03, 1.654830425502790e-02, 8.274152127513945e-03, -2.122072814762243e+01, -4.244145629524485e+01, -2.122072814762243e+01, 2.390376314408810e+01, 4.780752628817618e+01, 2.390376314408810e+01, 1.003762515443430e-03, 2.007525030932164e-03, 1.003762515443430e-03, 9.608422051934442e-06, 1.921687454876074e-05, 9.608422051934442e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([4.206027853927250e-02, 4.206027853927247e-02, 3.254560529290233e-02, 3.254560529290233e-02, -6.872875352606037e-03, -6.872875352606027e-03, -7.142028677565647e-02, -7.142028677564048e-02, -4.388785155893495e-02, -4.388785152868283e-02, -3.304028325594660e-08, -3.304028325658058e-08, -8.731347932230197e-20, -8.729893919252311e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
