
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lv_rpw86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.762250246967406e+00, -1.241864682711432e+00, -4.073163055143236e-01, -1.582882636182263e-01, -7.455575297422203e-02, -4.918352573946064e-02, -3.814340998249729e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lv_rpw86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.279073534166326e+00, -2.281190070092713e+00, -1.558080833422335e+00, -1.559463212284620e+00, -2.723224889488571e-01, -2.726804516994477e-01, -2.074343592266472e-01, -3.848088253570509e-02, -6.604147093119192e-02, -3.991677138757686e-03, -3.963880861634144e-02, -3.971677892849024e-02, -3.234305074551117e-03, -2.542679096333791e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lv_rpw86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lv_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.192078481277356e-04, 0.000000000000000e+00, -1.187798352149187e-04, -5.113536906547811e-04, 0.000000000000000e+00, -5.096201178245433e-04, -1.312689954009948e-01, 0.000000000000000e+00, -1.310111963721374e-01, -1.783664413891623e+00, 0.000000000000000e+00, -2.416628770248070e+02, -7.301304792194753e+01, 0.000000000000000e+00, -2.023706543600348e+06, -2.153750485425540e+02, 0.000000000000000e+00, -2.133507146154957e+02, -4.694867223786248e+06, 0.000000000000000e+00, -1.229067417938540e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
