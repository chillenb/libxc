
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lg93_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lg93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218572343964578e-01, -5.733528478240474e-01, -3.644667703304356e-01, -1.357017973699478e-01, -1.607190688899278e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lg93_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lg93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285111809842193e-01, 1.136780840192358e-16, -7.126621681119276e-01, -2.545967049038198e-16, -3.779217769835404e-01, -5.129185320289511e-17, -1.338274402361533e-01, -6.714978939435683e-17, -1.501555498459925e-02, -2.808271008216309e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lg93_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lg93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.690961453951650e-03, 0.000000000000000e+00, 0.000000000000000e+00, -2.406443118855859e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.359388989474750e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.302506884874030e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.125011647882275e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
