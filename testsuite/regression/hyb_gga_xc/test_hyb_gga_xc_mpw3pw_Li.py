
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw3pw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.510029308816357e+00, -1.084461690689655e+00, -3.467185752577929e-01, -1.471206212073345e-01, -6.912495518730202e-02, -4.803449665952787e-03, -1.249428666821993e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw3pw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.914481261077984e+00, -1.916031317450835e+00, -1.328829594971359e+00, -1.329805530355757e+00, -3.198567275728746e-01, -3.198472731648528e-01, -1.912358060041444e-01, -1.075954847971980e-01, -7.128461816447562e-02, 3.135572214446672e-01, -9.089059510898174e-03, -8.867358603920164e-03, -1.706662539432848e-04, -1.434698887707093e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw3pw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.664980637903508e-04, 6.453534174202383e-05, -1.658424989429407e-04, -6.380869189955935e-04, 2.118487154551634e-04, -6.357652353623929e-04, -7.033030671949370e-02, 6.140536786824862e-03, -7.026678029234716e-02, -5.664493015718725e-01, 5.282761836866798e+00, 2.727330603622557e+01, -4.307379840036948e+01, 1.915084822813345e+01, 3.414885321180523e+02, 2.469745751774144e+01, 2.805909535933161e-04, 2.320369841449651e+01, 2.756576637581507e+02, 2.603264411672309e-06, 4.228247336860049e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
