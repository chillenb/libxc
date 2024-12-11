
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.629965187898325e+00, -1.108016760821895e+00, -2.158754450828682e-01, -4.911444439417845e-02, -4.367546709469769e-03, -1.905359230562001e-05, -1.237691643044953e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.133552708379859e+00, -2.135585693476445e+00, -1.388034187871261e+00, -1.389299431091889e+00, -2.156208247608981e-01, -2.158720732830575e-01, -8.463761477062562e-02, -9.767402677860863e-02, -1.167510540025113e-02, 3.428185822404462e-01, -3.926113174661098e-05, -3.837267610990980e-05, -2.980101821010188e-10, -1.070704492883178e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.027172801839708e-04, 9.190971700708733e-05, -2.017954683801961e-04, -9.130772467578256e-04, 2.980993506782570e-04, -9.096494082515210e-04, -6.437061666998606e-02, 6.249948659585063e-03, -6.416506111031527e-02, 2.896935364538005e+00, 6.762268918356340e+00, 3.381134459178170e+00, 9.063588442050186e+00, 2.258698854598489e+01, 1.129349427299244e+01, 1.678587300264123e-04, 3.357174600576258e-04, 1.678587300264123e-04, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
