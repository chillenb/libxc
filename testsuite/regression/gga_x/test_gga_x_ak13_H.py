
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ak13_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ak13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.219255897392081e-01, -5.857898757298019e-01, -3.853908017552989e-01, -3.150528099690828e-01, -1.322736363193703e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ak13_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ak13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.283408031595337e-01, -2.443642448418615e-17, -6.940776457864565e-01, -2.667898398604158e-16, -3.406355838940933e-01, -4.445399599645040e-17, 1.315373409997818e-01, -1.385983302193890e-16, 5.229134654762471e-01, -2.302875390773431e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ak13_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ak13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.370231119023374e-02, 0.000000000000000e+00, 0.000000000000000e+00, -4.039925333052374e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.782983947578578e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.208894316266996e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.827141892527731e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
