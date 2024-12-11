
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_sg4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.195929345423695e-02, -1.492412887130328e-02, -7.692044669059209e-03, -1.083652056568749e-02, -1.569853905510373e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_sg4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.715204808453529e-02, 1.972973858472130e+00, -3.612415553723482e-02, 1.214308482333258e+02, -5.758460453005192e-03, 8.691621711432525e+01, 1.592469860323058e-03, 1.878672209478465e+01, -2.001690237919110e-03, -6.965489951221249e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_sg4_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.112809703278079e-02, 4.225619406556158e-02, 2.112809703278079e-02, 8.903923455218640e-03, 1.780784691043728e-02, 8.903923455218640e-03, -9.857233267437173e-03, -1.971446653487435e-02, -9.857233267437173e-03, -1.855743474600928e+00, -3.711486949201856e+00, -1.855743474600928e+00, -1.002473364999841e-106, -2.004946729999682e-106, -1.002473364999841e-106])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
