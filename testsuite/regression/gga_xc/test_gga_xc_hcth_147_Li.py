
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_147_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_147", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.916333163661312e+00, -1.346142241719279e+00, -4.650708651332033e-01, -1.762610141044590e-01, -9.084395164539627e-02, -1.860581662637797e-02, -4.156119024107774e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_147_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_147", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.560191488657951e+00, -2.562557119389862e+00, -1.741895606414821e+00, -1.743429203786129e+00, -3.558647815755653e-01, -3.566043943029624e-01, -2.384876150191269e-01, 6.762175334526572e-01, -6.252867072617330e-02, 4.418314151707795e-01, -2.605875537564583e-02, -2.499250071547309e-02, -8.610412566558443e-04, 3.041143218116916e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_147_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_147", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.581711647462882e-05, 0.000000000000000e+00, 3.612053723305628e-05, -1.921489035490622e-04, 0.000000000000000e+00, -1.903322066230315e-04, -1.271094106398471e-01, 0.000000000000000e+00, -1.267419603222325e-01, 2.166191184798222e+00, 0.000000000000000e+00, 1.507194857792274e+02, -1.246886322111470e+02, 0.000000000000000e+00, 1.805406806005185e+04, -8.489841767228057e-02, 0.000000000000000e+00, 1.550473026082134e-01, -5.852973231073526e+00, 0.000000000000000e+00, 3.496495543450298e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
