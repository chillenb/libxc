
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_gaploc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gaploc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.247068212624266e-02, -2.595263724563179e-02, -4.545634312484637e-03, -1.154125654603895e-03, -4.732562073224770e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_gaploc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gaploc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.644682239873463e-02, -2.240088326344734e-01, -7.822887578451133e-02, -2.630951902497231e-01, -2.222243416090218e-02, -7.804425921346886e-02, -2.423632702738347e-03, -1.782930878466095e-02, -1.729356459220601e-05, -9.920026077805281e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_gaploc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gaploc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.183705174242774e-08, 2.367410348485547e-08, 1.183705174242774e-08, 2.385098013656770e-02, 4.770196027313539e-02, 2.385098013656770e-02, 3.853268268157268e-02, 7.706536536314534e-02, 3.853268268157268e-02, 1.414298629996829e-01, 2.828597259993657e-01, 1.414298629996829e-01, 1.003408792544951e+01, 2.006817585089900e+01, 1.003408792544951e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
