
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220643816264014e-01, -5.790458137443317e-01, -3.623125246633661e-01, -1.418120618494147e-01, -8.065057857967080e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282424323197450e-01, -1.378964909205052e-17, -7.165970626524547e-01, -2.873504204051434e-16, -3.974156996640043e-01, -4.002651706958148e-17, -1.380097577115825e-01, -6.240144777938385e-17, -1.074405084831451e-02, -1.926797058701757e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.804970735837058e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.576247939072931e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.870924939371314e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.748773538483523e+00, 0.000000000000000e+00, 0.000000000000000e+00, -7.479176096546102e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
