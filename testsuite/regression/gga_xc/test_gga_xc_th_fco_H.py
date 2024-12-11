
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fco_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fco", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.590472154019898e-01, -8.195426139091448e-01, -5.884731872574822e-01, -3.245794666724635e-01, -2.295677814071943e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fco_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fco", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.056563282595191e+00, -8.892722259420860e-02, -9.217895832788534e-01, 4.797681559379635e-02, -5.992129209965658e-01, 7.767897308384808e-02, -2.011264584884868e-01, -3.014785711653468e-02, -7.092498812853071e-02, 1.292287373550068e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fco_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fco", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.524045293215996e-02, -2.128947220031682e-03, 7.950507611988547e+17, -4.042531481445146e-02, -4.921683860355164e-03, 5.778086873285884e+17, -3.144629203069065e-01, -1.015035876935139e-01, -4.582521510518506e+17, -2.331055044616324e+01, -1.220676875159593e+01, -2.520734979920994e+18, -1.780652237637470e+05, 6.762514950021640e+05, -4.617343533589172e+18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
