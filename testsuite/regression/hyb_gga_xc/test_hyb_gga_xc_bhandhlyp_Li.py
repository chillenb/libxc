
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_bhandhlyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandhlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.566145949719862e-01, -6.949146348860239e-01, -2.019676211999301e-01, -8.030765752182507e-02, -4.063122522463120e-02, -6.868102458984178e-02, -2.683703384754778e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_bhandhlyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandhlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.185068922564224e+00, -1.185961042101930e+00, -8.270273436575086e-01, -8.275390321269971e-01, -2.742285439002183e-01, -2.745903359797444e-01, -1.025575802609157e-01, -1.001392013172784e-01, -3.667336314931122e-02, -3.416159128558362e-02, -2.102010312342631e-02, -2.119921472076875e-02, -3.751415285308458e-03, -3.320157465378577e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_bhandhlyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandhlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.303828467040847e-04, 5.222815421851711e-06, -1.300538330166086e-04, -4.747230640426079e-04, 3.646941789248587e-05, -4.737639771539023e-04, -2.273062355685302e-02, 4.773762863586187e-02, -2.255535550531493e-02, -2.205121128003833e+00, 4.596134769453040e+00, -6.663463518027756e+02, -3.829758793256747e+01, 2.356939734329661e+01, -2.425149883430908e+07, -5.823925851803007e+02, 7.936097321777658e-02, -5.833218145075798e+02, -7.200011182853191e+07, 0.000000000000000e+00, -2.144805611914038e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
