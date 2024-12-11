
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ncapr_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncapr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.797446266883781e+00, -1.288693519404615e+00, -4.466412100644733e-01, -1.603200536364089e-01, -8.288906790899284e-02, -4.980311037881524e-01, -9.045452734273218e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ncapr_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncapr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.236114064990260e+00, -2.238255408673703e+00, -1.509510594889339e+00, -1.510882599165173e+00, -3.256119301229314e-01, -3.254179047573356e-01, -2.050021370256918e-01, 2.266394804912464e-01, -7.120513272265259e-02, 2.395754734386968e-01, 2.218675390144497e-01, 2.274735615194504e-01, 2.728317034898940e-01, 2.556765893732832e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ncapr_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ncapr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.732731993875743e-04, 0.000000000000000e+00, -2.723226714143269e-04, -1.096214792095537e-03, 0.000000000000000e+00, -1.092673535800814e-03, -1.308593426722334e-01, 0.000000000000000e+00, -1.308581954656609e-01, -4.187494325638792e+00, 0.000000000000000e+00, -8.617598816721325e+03, -8.602702989871838e+01, 0.000000000000000e+00, -9.467228162228242e+08, -7.307975129024378e+03, 0.000000000000000e+00, -7.402873528085129e+03, -3.243664172032427e+09, 0.000000000000000e+00, -1.038085927832704e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
