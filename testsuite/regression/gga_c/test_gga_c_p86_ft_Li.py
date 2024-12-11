
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86_ft_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.331938976238458e-02, -4.671643087095365e-02, 3.661159717066064e-03, -1.591271189964331e-02, -2.436273014941915e-03, -6.749577460215788e-03, -1.672862342604614e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86_ft_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.172348891132761e-01, -1.170761451792176e-01, -1.042958046524970e-01, -1.041764770190394e-01, -2.442504988419618e-02, -2.443795599903752e-02, -2.347771111920923e-02, -1.139515276187296e-01, -1.435761635607451e-02, -6.198429405861142e-02, -8.487071546414721e-03, -8.584808379014189e-03, -1.994023115233301e-04, -2.814219358193011e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86_ft_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86_ft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.362202616896609e-05, 8.724405233793217e-05, 4.362202616896609e-05, 1.467225641609766e-04, 2.934451283219532e-04, 1.467225641609766e-04, 6.714371754824413e-03, 1.342874350964883e-02, 6.714371754824413e-03, 2.622121334579462e+00, 5.244242669158924e+00, 2.622121334579462e+00, 2.918484088762568e+01, 5.836968177525135e+01, 2.918484088762568e+01, -4.772717840252149e-03, -9.545435680504298e-03, -4.772717840252149e-03, -1.025999751632920e-29, -2.051999503265839e-29, -1.025999751632920e-29])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
