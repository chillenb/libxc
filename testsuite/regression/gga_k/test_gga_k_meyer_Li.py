
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_meyer_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.633580987461021e+01, 8.105476547569589e+00, 8.231773564278445e-01, 1.328643978390465e-01, 2.859757762496560e-02, 3.063258465969247e+00, 1.356889842284533e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_meyer_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.594697181686555e+01, 2.599462075703023e+01, 1.227193325594381e+01, 1.229338611476317e+01, 2.811144965998508e-01, 2.787062509799378e-01, 2.138648601113620e-01, -3.089323462071737e+00, 2.502628785020967e-02, -1.210259755103958e+00, -3.067830618217266e+00, -3.173114689632209e+00, -1.418461529157469e+00, -1.185593855347997e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_meyer_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.151495785943900e-03, 0.000000000000000e+00, 2.145672032110285e-03, 6.459681534194482e-03, 0.000000000000000e+00, 6.443142714052148e-03, 5.287793975022195e-01, 0.000000000000000e+00, 5.297459791087971e-01, 2.902467089465332e+00, 0.000000000000000e+00, 7.829865630206406e+04, 4.953285131024326e+01, 0.000000000000000e+00, 2.454376723988761e+09, 6.733532616313123e+04, 0.000000000000000e+00, 6.882610962924840e+04, 8.236143164591826e+09, 0.000000000000000e+00, 2.292358354315005e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
