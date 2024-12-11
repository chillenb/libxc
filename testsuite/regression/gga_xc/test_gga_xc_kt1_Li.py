
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_kt1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.937765173617701e+00, -1.436282460965547e+00, -3.815253936389869e-01, -1.750066653985628e-01, -7.320189607125452e-02, -1.819289531765209e-02, -3.757080164547107e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_kt1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.274630301247799e+00, -2.276675738673147e+00, -1.510635497889452e+00, -1.511880913642426e+00, -4.379253027036150e-01, -4.378021191932222e-01, -2.302378400438677e-01, -1.434441391524872e-01, -9.606696465550921e-02, -7.484419423330232e-02, -2.379454075539524e-02, -2.377814747780557e-02, -4.986035015558134e-04, -5.016408535284593e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_kt1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.919594560353022e-04, 0.000000000000000e+00, -4.902048214802675e-04, -2.061349402651428e-03, 0.000000000000000e+00, -2.054586108301680e-03, -5.486396554400828e-02, 0.000000000000000e+00, -5.487195057375636e-02, -5.951829666719937e-02, 0.000000000000000e+00, -5.999998880484111e-02, -5.998800963135268e-02, 0.000000000000000e+00, -5.999999992242971e-02, -5.999998631077526e-02, 0.000000000000000e+00, -5.999998670470278e-02, -5.999999999999771e-02, 0.000000000000000e+00, -5.999999999999940e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
