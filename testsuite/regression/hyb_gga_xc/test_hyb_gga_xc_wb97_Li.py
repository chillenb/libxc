
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.635888033214207e+00, -1.110269226758732e+00, -2.326313114403896e-01, -3.997621359629919e-02, -6.258031285697171e-03, 4.949937775401438e-03, 6.752923500194761e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.147362022933823e+00, -2.149491800041373e+00, -1.377087464376355e+00, -1.378357864851603e+00, -1.058035315707080e-01, -1.062935171842612e-01, -9.632712825670155e-02, 2.523691262970875e-01, 3.998940871476365e-04, 1.468546232866022e-01, 6.261568921954613e-03, 6.528502382921529e-03, 1.120516559675963e-05, 3.113764941297423e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.551739217030174e-04, 0.000000000000000e+00, -1.543168120874601e-04, -8.556849976048710e-04, 0.000000000000000e+00, -8.525320764073664e-04, -1.387678546267536e-01, 0.000000000000000e+00, -1.385672856442156e-01, 1.348554960275687e+01, 0.000000000000000e+00, -5.000935678291204e+01, -2.399882550884696e+01, 0.000000000000000e+00, -5.978533534485883e+03, -6.384325668782749e-01, 0.000000000000000e+00, -6.733721796570926e-01, -1.194705281561345e+00, 0.000000000000000e+00, -1.605722641689816e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
