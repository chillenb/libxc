
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.131437108792289e+00, -1.508011558981075e+00, -6.849127017116712e-01, -3.586610597245229e-01, -2.681015234438642e-01, -3.407475746968527e-01, 1.615039880816763e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.926817043148511e+00, -2.929313297159362e+00, -1.954834709497309e+00, -1.956389852283069e+00, -4.288520835837963e-01, -4.285957855238133e-01, -3.973036561745414e-01, -7.748239165769184e-02, -2.313077436407880e-01, 6.802159950221834e-02, -3.594262920053592e-01, -3.565499030384595e-01, -2.090282662687657e+00, -3.228576670727889e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.001232436103249e-04, 0.000000000000000e+00, 1.986788655237325e-04, 1.552489782077481e-05, 0.000000000000000e+00, 1.157084705087153e-05, -1.983964254408679e-01, 0.000000000000000e+00, -1.983869275123862e-01, -1.413706098206514e+01, 0.000000000000000e+00, -2.754448276789479e+03, -1.924198497961688e+02, 0.000000000000000e+00, -2.396733487452496e+08, -6.062666512990045e+02, 0.000000000000000e+00, -6.295707000415310e+02, 1.318651390555750e+10, 0.000000000000000e+00, 1.086315368957952e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
