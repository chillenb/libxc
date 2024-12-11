
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_rational_p_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_rational_p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.410233887326312e+01, 6.083255590394719e+00, 7.547267559006712e-02, 1.208862118699579e-01, 6.786327334108792e-03, 1.322460635036883e-08, 1.143812643129050e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_rational_p_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_rational_p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.774164883896902e+01, 2.779021583128489e+01, 1.357061549625220e+01, 1.359364061539236e+01, 3.352763936972744e-01, 3.339082415296606e-01, 2.235260794852125e-01, 6.004777235460299e-08, 2.541342912268605e-02, 7.748340846490330e-15, 7.815240171882974e-08, 7.157818358623506e-08, 8.117339523405287e-16, 1.928868847052196e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_rational_p_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_rational_p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.302304091461348e-03, 0.000000000000000e+00, -7.285109324365600e-03, -1.813228219827224e-02, 0.000000000000000e+00, -1.809356656995508e-02, -1.014681271483055e-01, 0.000000000000000e+00, -1.009267212447776e-01, -1.069605484394438e+01, 0.000000000000000e+00, -4.070262781244526e-04, -3.086038849145887e+01, 0.000000000000000e+00, -4.158857126994243e-06, -4.593041849503710e-04, 0.000000000000000e+00, -4.154871489719638e-04, -1.247636198064859e-06, 0.000000000000000e+00, -9.872234847938722e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
