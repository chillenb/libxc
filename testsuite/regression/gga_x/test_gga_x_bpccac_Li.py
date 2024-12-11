
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bpccac_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bpccac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.795240010480794e+00, -1.285839299565168e+00, -4.436487040311519e-01, -1.600449833666756e-01, -8.313828951322282e-02, -3.150226553409878e-04, -4.422816966138960e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bpccac_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bpccac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.239364368470063e+00, -2.241506324876005e+00, -1.510834136467844e+00, -1.512213201995374e+00, -3.954791973617666e-01, -3.967960318018027e-01, -2.052057860099571e-01, -1.125679186039284e-03, -7.037318839734136e-02, -8.228053776010208e-08, -1.325445299968643e-03, -1.250557246656183e-03, -2.085895355344873e-08, -8.956448704561330e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bpccac_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bpccac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.626600472841022e-04, 0.000000000000000e+00, -2.617419958417334e-04, -1.069193284452579e-03, 0.000000000000000e+00, -1.065696873468321e-03, -9.506811125972764e-02, 0.000000000000000e+00, -9.432792780337099e-02, -4.018387572489558e+00, 0.000000000000000e+00, 7.285061303611580e+00, -8.857503253639709e+01, 0.000000000000000e+00, 4.175689889070174e+01, 7.439935701029313e+00, 0.000000000000000e+00, 6.932018228464530e+00, 3.029810382345242e+01, 0.000000000000000e+00, 4.331544084864267e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
