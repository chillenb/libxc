
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_ncap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.859782300358997e+00, -1.333916990646094e+00, -4.449505098788209e-01, -1.762263706144553e-01, -8.521769187657077e-02, -6.193023449003521e-01, -1.181656451644269e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_ncap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.355185058624242e+00, -2.357166262679840e+00, -1.615852239479511e+00, -1.617105858985566e+00, -3.366473427878920e-01, -3.364081087865184e-01, -2.285749297488451e-01, 1.934129054318971e-01, -8.460450845828500e-02, 2.620984269228656e-01, 2.927123467871572e-01, 3.000799025756172e-01, 3.694844210790968e-01, 3.464484251502075e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_ncap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_ncap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.242695310329985e-04, 8.727067477050367e-05, -2.233386395402131e-04, -9.282800419735018e-04, 2.935567638936289e-04, -9.248024920140937e-04, -1.319168850888814e-01, 1.344716138288092e-02, -1.319449923006896e-01, -1.489792573492386e+00, 5.244660104550263e+00, -1.088292527136151e+04, -5.862436888665226e+01, 5.839100514237006e+01, -1.240161814167629e+09, -9.221345839061780e+03, -9.583344691552998e-03, -9.345636394746532e+03, -4.263928288392473e+09, -2.099357782136131e-29, -1.366889884785422e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
