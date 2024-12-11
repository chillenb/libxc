
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_sa_tpss_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.042740068742787e-01, -5.618322525642848e-01, -3.260643685074811e-01, -8.988284335738780e-02, -4.251025328404654e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_sa_tpss_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.293290098179327e-01, -6.258162721822941e-17, -7.546395286529219e-01, -2.229885395478387e-16, -4.363668922326476e-01, -1.357049952824082e-17, -1.192583400666043e-01, -4.969308779576313e-17, -5.667832278866467e-03, -6.382004185867456e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.484450100214293e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.040958369289738e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.354065517036555e-01, 0.000000000000000e+00, 0.000000000000000e+00, -3.715326528348214e+00, 0.000000000000000e+00, 0.000000000000000e+00, -4.320037094952603e+05, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.069140521883364e+01, 0.000000000000000e+00, 7.065766256712452e-03, 0.000000000000000e+00, 1.940688151001311e-03, 0.000000000000000e+00, -6.908651384403387e-04, 0.000000000000000e+00, -1.765479973830373e-07, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
