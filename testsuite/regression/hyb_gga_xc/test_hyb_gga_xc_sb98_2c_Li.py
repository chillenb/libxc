
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_2c_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.474802939001600e+00, -1.055206054723001e+00, -3.463134446752906e-01, -1.369895683822712e-01, -6.569121600779705e-02, -1.279193665667471e-02, -2.514529915241072e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_2c_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.883667649290057e+00, -1.885178518392300e+00, -1.302303418930260e+00, -1.303281975006618e+00, -3.104593174835292e-01, -3.109645089026358e-01, -1.760579588748326e-01, 3.536494007978661e-01, -5.370912612620481e-02, 2.313449845021880e-01, -1.787501570965522e-02, -1.727586378163843e-02, -5.010527712838879e-04, 1.247399072147528e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_2c_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.101299826531964e-04, 0.000000000000000e+00, -1.097476991631427e-04, -4.733481889003252e-04, 0.000000000000000e+00, -4.716227166159476e-04, -7.390295480866906e-02, 0.000000000000000e+00, -7.364943752097818e-02, -2.341469015937415e+00, 0.000000000000000e+00, 5.460162348577725e+01, -7.349524434088360e+01, 0.000000000000000e+00, 6.523322272635554e+03, -1.344219279295334e-01, 0.000000000000000e+00, -4.055891500106548e-02, -2.561470955699683e+00, 0.000000000000000e+00, 1.199719726739532e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
