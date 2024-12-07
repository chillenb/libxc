
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.331427216601507e-02, -4.670901392822047e-02, 3.681689640734326e-03, -1.591259549341341e-02, -2.434590245040283e-03, -6.749577227957240e-03, -1.672862342604614e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.172433755574666e-01, -1.170846393050902e-01, -1.043069337293265e-01, -1.041876166128413e-01, -2.443704832233787e-02, -2.444993988276190e-02, -2.347797155394097e-02, -1.139516914820725e-01, -1.436002570840173e-02, -6.198530114554456e-02, -8.487067097042656e-03, -8.584803926816721e-03, -1.994023115233301e-04, -2.814219358193011e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.363533738525183e-05, 8.727067477050367e-05, 4.363533738525183e-05, 1.467783819468145e-04, 2.935567638936289e-04, 1.467783819468145e-04, 6.723580691440459e-03, 1.344716138288092e-02, 6.723580691440459e-03, 2.622330052275132e+00, 5.244660104550263e+00, 2.622330052275132e+00, 2.919550257118503e+01, 5.839100514237006e+01, 2.919550257118503e+01, -4.791672345776499e-03, -9.583344691552998e-03, -4.791672345776499e-03, -1.049678891068065e-29, -2.099357782136131e-29, -1.049678891068065e-29]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
