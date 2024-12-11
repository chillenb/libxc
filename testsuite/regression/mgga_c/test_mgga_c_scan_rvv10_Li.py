
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_scan_rvv10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.033882340639840e-02, -5.615699610386440e-02, -6.498879409859544e-02, -2.276512352372785e-03, -1.518603161346363e-02, -2.016493942151557e-04, -2.613740116899876e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_scan_rvv10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.472978064836610e-02, -1.458636053099015e-02, 1.224576080450235e-02, 1.247632299173077e-02, -6.820947134485039e-02, -6.826684027960464e-02, 1.584075756800089e-03, -1.573131829747014e-01, -1.229191849437275e-02, -5.721243928705695e-02, -4.042546495325853e-04, -4.112299306096803e-04, -1.669290497042180e-08, -1.594395420616185e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_rvv10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.795885387810238e-05, 1.359177077562048e-04, 6.795885387810238e-05, 6.180616594158535e-04, 1.236123318831707e-03, 6.180616594158535e-04, 2.484376422268863e-01, 4.968752844537726e-01, 2.484376422268863e-01, 3.380978779794348e+00, 6.761957559588694e+00, 3.380978779794348e+00, 1.900897526226118e+02, 3.801795052452238e+02, 1.900897526226118e+02, 2.682000845247828e-02, 5.364001690495657e-02, 2.682000845247828e-02, 2.041857137021838e-04, 4.083714274043677e-04, 2.041857137021838e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_rvv10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.975119482602264e-03, -3.975119482602262e-03, -1.073213959607764e-02, -1.073213959607764e-02, -1.278855002579240e-03, -1.278855002579239e-03, -1.134449257692161e-01, -1.134449257691910e-01, -4.912667931254514e-02, -4.912667927285537e-02, -1.340109833902018e-07, -1.340109833901994e-07, -7.987416244970234e-24, -7.987475476220700e-24])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
