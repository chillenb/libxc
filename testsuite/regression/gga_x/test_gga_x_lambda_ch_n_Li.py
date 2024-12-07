
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_ch_n_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_ch_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.794242943954321e+00, -1.283218480838326e+00, -4.120054802666119e-01, -1.600197349337648e-01, -8.010689340112320e-02, -2.002221612797831e-02, -3.740806723063323e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_ch_n_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_ch_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.243166302560005e+00, -2.245298665919016e+00, -1.520347417043728e+00, -1.521714889384624e+00, -4.050488080565874e-01, -4.052179817931738e-01, -2.053283023581254e-01, -2.545419697272325e-02, -7.719925181538340e-02, -8.085100963181011e-04, -2.676216858419247e-02, -2.656966290903966e-02, -5.400395879124835e-04, -3.839190295934542e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_ch_n_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_ch_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.538639670011541e-04, 0.000000000000000e+00, -2.529917920304839e-04, -1.000652235030682e-03, 0.000000000000000e+00, -9.974520018040836e-04, -6.998036778234840e-02, 0.000000000000000e+00, -6.979986365661635e-02, -3.938321844797755e+00, 0.000000000000000e+00, -2.469125275158406e-01, -6.479348074960957e+01, 0.000000000000000e+00, -1.579187093304862e+00, -2.509198066796780e-01, 0.000000000000000e+00, -2.343130800025671e-01, -1.149589726062295e+00, 0.000000000000000e+00, -1.645519328864087e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
