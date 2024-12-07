
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw3_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.035431504282009e+00, 1.697672169438623e+00, 6.199790550683338e-01, 6.271015009451379e-02, 1.632130597498568e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw3_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.388588702129613e+00, 5.091572729257148e-16, 2.670619895077082e+00, 4.574038248911790e-16, 8.925605457608384e-01, 4.862430947712147e-17, 8.490294032687810e-02, 3.809030504741147e-17, 2.718658073122418e-04, 4.986799267998220e-22]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw3_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.824315524807195e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.377666246088949e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.073621810749941e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.207748282120573e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.246259300072178e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
