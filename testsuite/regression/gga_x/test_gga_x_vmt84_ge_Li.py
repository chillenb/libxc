
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt84_ge_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.769673663079482e+00, -1.251405228352217e+00, -3.907926990937847e-01, -1.586803319002164e-01, -7.507277447926795e-02, -1.067938103011802e-02, -6.423415171602323e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt84_ge_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.270821463469447e+00, -2.272941314631810e+00, -1.549766792859579e+00, -1.551144532921085e+00, -3.459614810345751e-01, -3.459540449836569e-01, -2.069469518497140e-01, -1.528549627328438e-02, -7.292364594809360e-02, -1.189920777758582e-09, -1.597419028994126e-02, -1.590169553611726e-02, -3.025278567093242e-10, -1.300428873720614e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt84_ge_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.503161852158242e-04, 0.000000000000000e+00, -1.497862111675490e-04, -6.221756144986349e-04, 0.000000000000000e+00, -6.201175229931531e-04, -8.490481681939091e-02, 0.000000000000000e+00, -8.481889363917273e-02, -2.283391513077731e+00, 0.000000000000000e+00, 1.697270517482461e+01, -5.946173396776283e+01, 0.000000000000000e+00, 6.033142343008679e-01, 1.384938411455763e+01, 0.000000000000000e+00, 1.427904279403330e+01, 4.391532728352826e-01, 0.000000000000000e+00, 6.286021305296537e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
