
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_rcam_b3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.356318707497691e+00, -9.302808542840234e-01, -1.229189356826774e-01, -2.051774758758344e-02, 4.424168469298676e-03, -8.997423387591278e-04, -1.399810354267851e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_rcam_b3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.726443308631631e+00, -1.727987457662278e+00, -1.151180605459508e+00, -1.152098435920014e+00, -2.843044487336446e-01, -2.846853594269301e-01, -4.392554039471777e-02, -8.039475174447297e-02, 3.724112148418059e-03, -3.025697204709918e-02, -7.609693039596997e-04, -8.664740996770410e-04, 1.624159257010869e-06, -8.064444645067826e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_rcam_b3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_rcam_b3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.138763512131939e-04, 5.222815421851711e-06, -2.133066280316730e-04, -7.240112026833657e-04, 3.646941789248587e-05, -7.224469477825693e-04, 8.279097814677162e-03, 4.773762863586187e-02, 8.480173786303679e-03, -4.310279442645325e-01, 4.596134769453040e+00, -2.104270141699427e-02, -4.404241511650330e-01, 2.356939734329661e+01, -1.256051784788971e+05, -2.976197011279968e+00, 7.936097321777658e-02, -2.980797647530944e+00, -3.729605792717955e+05, 0.000000000000000e+00, -1.111009306971465e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
