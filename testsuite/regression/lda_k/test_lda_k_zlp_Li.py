
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_k_zlp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.812053363974602e+01, 1.364387500852475e+01, 7.967205796008974e-01, 2.349565179936713e-01, 3.699081548701773e-02, 1.241859048255428e-03, 4.413928169730032e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_k_zlp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_k_zlp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([4.663090448303736e+01, 4.671532419404976e+01, 2.265000261920997e+01, 2.268869572024455e+01, 1.327166069095733e+00, 1.326036420103387e+00, 3.915417216282520e-01, 1.698716450393077e-03, 6.163852389439081e-02, -1.095106508357514e-05, 2.084588988055598e-03, 2.054374785151195e-03, 8.462941099709422e-07, 4.277089764314738e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
