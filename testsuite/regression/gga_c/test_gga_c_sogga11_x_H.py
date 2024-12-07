
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_sogga11_x_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.207557003161048e-02, -2.585194047732327e-02, -2.460285944071908e-02, -5.382688060999685e-02, -5.670574802450799e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_sogga11_x_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.693404946488792e-02, 7.571885889497952e+00, -1.943381310144711e-02, -7.301061618646635e+01, -2.216297463003405e-02, -2.600219049421014e+01, -3.847461789233991e-02, -2.933364558301223e+01, -7.264405995919293e-03, -2.381281291331201e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_sogga11_x_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.550143484640291e-02, 3.100286969280583e-02, 1.550143484640291e-02, -4.865434941405757e-03, -9.730869882811514e-03, -4.865434941405757e-03, -1.387825913670771e-02, -2.775651827341542e-02, -1.387825913670771e-02, -3.012285439040219e+00, -6.024570878080438e+00, -3.012285439040219e+00, 2.758945723776659e+01, 5.517891447553318e+01, 2.758945723776659e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
