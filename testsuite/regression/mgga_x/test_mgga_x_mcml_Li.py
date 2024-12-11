
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mcml_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.856814815141000e+00, -1.249975436666831e+00, -2.799159882676440e-01, -1.658315133170946e-01, -6.054242630422493e-02, -1.146238512484021e-02, -2.122256108690134e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mcml_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.453298363837249e+00, -2.455045218345186e+00, -1.872013162075849e+00, -1.873424925749902e+00, -3.749396899560518e-01, -3.748940906231580e-01, -2.126548065118014e-01, -1.461282657547054e-02, -8.219553520178637e-02, -4.622009830254194e-04, -1.533330589425931e-02, -1.525697145754277e-02, 6.387137323144048e+03, -2.108479124258048e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mcml_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.536294740370800e-04, 0.000000000000000e+00, -1.519781271710864e-04, -2.065084272140636e-03, 0.000000000000000e+00, -2.055452202184324e-03, -2.929835515614838e-01, 0.000000000000000e+00, -2.933510659717162e-01, 1.089755901457646e+00, 0.000000000000000e+00, 2.109212165813075e-01, -1.345826130280099e+02, 0.000000000000000e+00, -4.752143252702795e+02, 7.783863964454188e-05, 0.000000000000000e+00, 2.020583723405953e-01, -2.936649769529674e+08, 0.000000000000000e+00, -2.889622287664852e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mcml_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.023587629017960e-03, -3.097190376360163e-03, 3.015849699518557e-02, 3.012134235216896e-02, 3.007578310742484e-04, 3.233618499922603e-04, -2.424626594337699e-01, 8.308842113909050e-12, 1.232740862255159e-02, 1.940822823031599e-07, -2.713427674472172e-11, 2.620517063602107e-12, 3.565564279938416e-02, 3.812346968433121e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
