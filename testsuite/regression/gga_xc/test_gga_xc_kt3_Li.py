
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_kt3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.020704125070933e+00, -1.470059594957600e+00, -4.408725238531336e-01, -1.715487936433575e-01, -7.964920469662970e-02, -2.555182594934530e-02, -4.699444446028700e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_kt3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.467637519033662e+00, -2.469897547240690e+00, -1.638700977204163e+00, -1.640108581744473e+00, -4.298055586459170e-01, -4.303207110319184e-01, -2.278967989777178e-01, -1.012552693281234e-01, -7.092393630384591e-02, -2.715071152558681e-02, -3.404533823029288e-02, -3.389779879228524e-02, -6.589252696930156e-04, -5.365720658954265e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_kt3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.673632098564372e-04, 4.514648655987416e-06, -3.660701708005600e-04, -1.671577955672380e-03, 3.152449305102582e-05, -1.666239417707061e-03, -1.150531054155653e-01, 4.126483583149672e-02, -1.147553338424480e-01, -4.606406746244529e-01, 3.972940259928133e+00, 2.426638540978493e+00, -7.720994878678013e+01, 2.037359918812168e+01, 1.194929397760428e+01, -5.273093746607742e-01, 6.860033949820503e-02, -4.925496411582544e-01, -2.435660192602990e+00, 0.000000000000000e+00, -3.469141498853202e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
