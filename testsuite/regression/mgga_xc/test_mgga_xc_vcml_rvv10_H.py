
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_vcml_rvv10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.798052096595714e-01, -6.188960936263025e-01, -3.794204710950831e-01, -1.244540780207781e-01, -4.103046651814973e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_vcml_rvv10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.994380828871306e-01, 1.721624345323563e+01, -7.929858033223149e-01, 7.103853482010346e+02, -4.549008750805215e-01, 4.161403937222119e+02, -1.754759922527588e-01, 5.429911640212540e+00, -5.499147537289003e-03, 4.283130421726685e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.776762197760184e-03, 3.089137037709527e-02, 1.544568518854764e-02, -1.248007289965489e-02, 1.998028342303394e-02, 9.990141711516970e-03, -1.050744821546636e-01, 8.638774609755090e-02, 4.319387304877545e-02, 1.085466138004571e+00, 2.814219005341180e-01, 1.407109502670590e-01, 1.445675682433039e+01, 5.920208480833401e-03, 2.960104239720123e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.101633566660515e-12, 0.000000000000000e+00, -4.637777264895696e-11, 0.000000000000000e+00, -2.327003352854420e-10, 0.000000000000000e+00, -1.155765514414988e-15, 0.000000000000000e+00, 9.045008128315109e-06, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
