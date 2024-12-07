
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_cam_s12g_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.694855826364666e-01, -4.173072071601566e-01, -2.463403805160121e-01, -9.063922355424964e-02, -4.720341562661641e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_cam_s12g_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.487103735134837e-01, -1.556724826700601e-17, -5.574681403950412e-01, -1.944126858287891e-16, -2.618426768071682e-01, -5.383422690659638e-17, -1.006836937884903e-01, -4.976905519759007e-17, -6.291200882196404e-03, -2.429297299401481e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_cam_s12g_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_cam_s12g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.662981715008165e-05, 0.000000000000000e+00, 0.000000000000000e+00, -8.126684832526139e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.573433073033750e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.284353738083786e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.078604587161017e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
