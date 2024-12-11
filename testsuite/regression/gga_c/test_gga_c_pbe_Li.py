
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.998883374765556e-02, -4.346042522812144e-02, -3.078654696611095e-03, -1.516244914509290e-02, -1.348142595314867e-03, -7.609699831205574e-09, -1.790559887859811e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.160122996920183e-01, -1.158864716006442e-01, -1.014563140995033e-01, -1.013618241399399e-01, -1.592623935402938e-02, -1.593189730306958e-02, -2.421895779126653e-02, -9.764053356841586e-02, -6.581997633501091e-03, 3.428185832405125e-01, -4.924935602294473e-08, -4.949675553441214e-08, -1.135871182268017e-15, -1.343631423570552e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.595485850354368e-05, 9.190971700708733e-05, 4.595485850354368e-05, 1.490496753391285e-04, 2.980993506782570e-04, 1.490496753391285e-04, 3.124974329792531e-03, 6.249948659585063e-03, 3.124974329792531e-03, 3.381134459178170e+00, 6.762268918356340e+00, 3.381134459178170e+00, 1.129349427299244e+01, 2.258698854598489e+01, 1.129349427299244e+01, 1.678587300264123e-04, 3.357174600576258e-04, 1.678587300264123e-04, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
