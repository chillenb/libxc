
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_fr_pw86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.636341365580267e+01, 8.166431481103384e+00, 6.285860878715159e-01, 1.318090282161616e-01, 2.640619425395839e-02, 2.986314289362507e-03, 4.390617505891872e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_fr_pw86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.573575372548720e+01, 2.578343921430445e+01, 1.216335340987061e+01, 1.218438926094229e+01, 8.215307436811327e-01, 8.211935794351135e-01, 2.130853674057072e-01, 3.142768511710906e-03, 3.508120940456858e-02, 1.040769949776572e-05, 3.401822482289470e-03, 3.384852527772115e-03, 5.632797699535051e-06, 3.148108281503665e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_fr_pw86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.592456170471376e-03, 0.000000000000000e+00, 2.585030536529610e-03, 7.566882946784983e-03, 0.000000000000000e+00, 7.549226097437950e-03, 1.097636999742559e-01, 0.000000000000000e+00, 1.095708089807389e-01, 3.244479928834965e+00, 0.000000000000000e+00, 1.409707082160355e+01, 1.953899750109577e+01, 0.000000000000000e+00, 3.724712956031049e+03, 1.321894791855036e+01, 0.000000000000000e+00, 1.299607379853325e+01, 5.771703231171568e+03, 0.000000000000000e+00, 1.074160734360765e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
