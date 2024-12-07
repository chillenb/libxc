
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ecmv92_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ecmv92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.222445782032932e-01, -5.796889113202566e-01, -3.616562546484283e-01, -1.430830958655879e-01, -8.357033666460520e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ecmv92_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ecmv92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.280448769564933e-01, -1.291321331998394e-17, -7.198151618265479e-01, -2.365737854950310e-16, -3.998335491537138e-01, -8.235862224147736e-17, -1.340893702899648e-01, -7.331125173137715e-17, -1.113121395474752e-02, -9.687795660086421e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ecmv92_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ecmv92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.476525689682157e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.466598788104805e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.799010710721021e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.380808870597034e+00, 0.000000000000000e+00, 0.000000000000000e+00, -9.187586258109205e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
