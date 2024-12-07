
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_w94_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_w94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.763179102779988e-02, -4.877603166673589e-02, -1.830299385566309e-03, -1.823848077009190e-03, -1.947009614433616e-06, -6.700318988741308e-08, -5.552208074865429e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_w94_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_w94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.214741818610263e-01, -1.200966691751340e-01, -1.334510026331485e-01, -1.324945411019749e-01, -9.389262656312803e-03, -9.411871221030625e-03, -2.167247995312908e-03, -2.743393962643380e+00, -8.680342602001031e-06, -5.714076612832222e+00, -3.486257335636175e-07, -3.541350328794353e-07, -2.704936054318469e-12, -3.489340035188998e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_w94_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_w94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.509861859708499e-05, 9.019723719416998e-05, 4.509861859708499e-05, 2.204782472437692e-04, 4.409564944875384e-04, 2.204782472437692e-04, 1.832760365912255e-03, 3.665520731824508e-03, 1.832760365912255e-03, 5.408338455919043e-01, 1.081667691183809e+00, 5.408338455919043e-01, 1.675274974678010e-02, 3.350549949356020e-02, 1.675274974678010e-02, 1.176251340579366e-03, 2.352502681158732e-03, 1.176251340579366e-03, 3.951715880537058e-03, 7.903431761074116e-03, 3.951715880537058e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
