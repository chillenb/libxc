
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lrc_wpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.687891647394192e+00, -1.160945139040140e+00, -2.595528982725687e-01, -6.488686908818239e-02, -6.828766818131004e-03, -3.259303628253769e-05, -2.200284682250894e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lrc_wpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.195187079214975e+00, -2.197190760079069e+00, -1.458881466000548e+00, -1.460150271909830e+00, -2.588097552788866e-01, -2.589741677865763e-01, -1.077558269130071e-01, -9.769750532174630e-02, -1.477407668338318e-02, 3.428185814627344e-01, -6.657872693130435e-05, -6.509534763900885e-05, -5.297789629611102e-10, -1.903369067051614e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lrc_wpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.014424606638627e-04, 9.190971700708733e-05, -2.005914024216367e-04, -8.279513322014127e-04, 2.980993506782570e-04, -8.248392341659397e-04, -6.780667773088832e-02, 6.249948659585063e-03, -6.763693665729502e-02, 2.414789762773676e+00, 6.762268918356340e+00, 3.381088356659774e+00, 4.960144529492871e+00, 2.258698854598489e+01, 1.129349427296008e+01, 1.060267398083148e-04, 3.357174600576258e-04, 1.123859901195476e-04, 1.606538923866599e-06, 3.212885779437900e-06, 1.606541887748125e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
