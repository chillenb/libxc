
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_n12_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.728001016998725e-02, -1.908076792997202e-02, -1.849176027541976e-02, -2.405763888672095e-02, -3.091287044990742e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_n12_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.114663640618404e-02, -2.592554956565175e-01, -4.948366935057374e-02, -2.677621744618666e-01, 1.280702628150858e-02, -2.154181032549127e-01, -2.320580329858287e-02, 5.197767416728657e-02, -3.940601248152400e-03, -9.171826683410030e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_n12_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.614566791823412e-01, 0.000000000000000e+00, -5.162356569791195e+20, 1.301123683153181e-02, 0.000000000000000e+00, -2.438665931884668e+20, -7.402593853835247e-02, 0.000000000000000e+00, 9.948499706272249e+19, -5.945443012539481e-01, 0.000000000000000e+00, -4.145471509064642e+19, -9.073151057951716e-01, 0.000000000000000e+00, -4.733952494306066e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
