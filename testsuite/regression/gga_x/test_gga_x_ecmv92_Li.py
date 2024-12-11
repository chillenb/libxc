
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ecmv92_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ecmv92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.801019466529915e+00, -1.289585084964473e+00, -4.345300698639006e-01, -1.605326523980977e-01, -8.234617222008203e-02, -2.320936793406457e-02, -4.337724561858392e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ecmv92_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ecmv92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.246012431653613e+00, -2.248149570797959e+00, -1.519180916919586e+00, -1.520558627825805e+00, -3.800403413163977e-01, -3.802311185019944e-01, -2.055464793674873e-01, -2.948864944978822e-02, -7.317403174608891e-02, -9.375215228661707e-04, -3.100064815381055e-02, -3.077916421560752e-02, -6.262129277442494e-04, -4.451805942212207e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ecmv92_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ecmv92", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.644677035740713e-04, 0.000000000000000e+00, -2.635632559636847e-04, -1.051515371094067e-03, 0.000000000000000e+00, -1.048113098714989e-03, -9.665641089811376e-02, 0.000000000000000e+00, -9.646486778999702e-02, -4.164383276153733e+00, 0.000000000000000e+00, -4.607473103850587e-01, -8.013501836726653e+01, 0.000000000000000e+00, -2.949357501973338e+00, -4.681786464571810e-01, 0.000000000000000e+00, -4.372130488410394e-01, -2.147025967126902e+00, 0.000000000000000e+00, -3.073247629892430e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
