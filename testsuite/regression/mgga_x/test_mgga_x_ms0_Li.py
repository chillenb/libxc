
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.933682547207351e+00, -1.258744633489178e+00, -2.499348838642552e-01, -1.777209473287758e-01, -5.540133895740799e-02, -1.469835120185761e-02, -2.556820389059564e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.745757496643298e+00, -2.748279530852887e+00, -1.938674476501854e+00, -1.940887670776717e+00, -3.408846400104321e-01, -3.412692252997148e-01, -2.431431044788317e-01, -1.869577324389206e-02, -8.066684673416394e-02, -5.932610538831557e-04, -1.967018471149702e-02, -1.951624476315872e-02, -3.962648448411552e-04, -1.868390259346569e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.277633704520573e-04, 0.000000000000000e+00, -5.255844489478079e-04, -2.718735999752757e-03, 0.000000000000000e+00, -2.714193822167747e-03, -3.504739653663594e-01, 0.000000000000000e+00, -3.510912718150148e-01, -5.258181660741449e+00, 0.000000000000000e+00, -6.420005604894172e-02, -1.720492401132811e+02, 0.000000000000000e+00, -4.109471059815212e-01, -2.737322023596283e-05, 0.000000000000000e+00, -6.092093809522077e-02, -2.762037957326081e-08, 0.000000000000000e+00, -3.442899270449117e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.405369798000680e-02, 2.401918530805219e-02, 3.820329864914065e-02, 3.827215803799080e-02, 1.388569972604416e-03, 1.493652157610847e-03, 1.738038974366123e-01, 6.370438933669128e-18, 5.692264325357579e-02, 9.239466110955389e-19, -1.040770193746186e-20, 2.270021881361352e-18, 3.351280085077397e-18, 1.712362528942261e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
