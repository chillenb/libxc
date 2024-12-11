
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_r_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.769589989384020e+00, -1.251192267202019e+00, -3.888870643517374e-01, -1.586867479623895e-01, -7.484603834944548e-02, -3.909517616953358e-02, -2.370193167298545e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_r_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.271125771185247e+00, -2.273244864155112e+00, -1.550522817493592e+00, -1.551899641444662e+00, -3.448332043984774e-01, -3.447831647292479e-01, -2.069568986063519e-01, -3.590186147011892e-02, -7.334186259706046e-02, -2.717515462816901e-03, -3.725190125244105e-02, -3.720360269773070e-02, -2.136497771214167e-03, -1.656521447016467e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_r_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.496033066017525e-04, 0.000000000000000e+00, -1.490770118052974e-04, -6.166957321330948e-04, 0.000000000000000e+00, -6.146612432012108e-04, -8.422001498305540e-02, 0.000000000000000e+00, -8.415512128192026e-02, -2.276751957260260e+00, 0.000000000000000e+00, -1.388257122796813e+02, -5.788504162766223e+01, 0.000000000000000e+00, -1.144669132107058e+06, -1.239399564961875e+02, 0.000000000000000e+00, -1.226762255303828e+02, -2.655491384049292e+06, 0.000000000000000e+00, -6.951758087411629e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
