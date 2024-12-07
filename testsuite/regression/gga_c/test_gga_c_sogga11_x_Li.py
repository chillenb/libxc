
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_sogga11_x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.707841932456885e-02, -7.605405457785236e-02, -1.414553766915461e-01, -1.596437131939213e-02, -1.265485284421962e-02, -2.457903395587035e-02, -6.061073244962802e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_sogga11_x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.987843653938781e-02, -5.975647128630518e-02, -3.754653564983244e-02, -3.744131419766075e-02, 1.750629943957818e-01, 1.750196260202288e-01, -2.226123148432532e-02, -1.086804922026007e-01, 1.132047053382777e-02, -1.251176163782563e+00, -3.128029001004510e-02, -3.162833017540621e-02, -7.130327823250006e-04, -1.046272261245774e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_sogga11_x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.323213441423207e-05, -4.646426882846414e-05, -2.323213441423207e-05, -1.338276501358928e-04, -2.676553002717857e-04, -1.338276501358928e-04, -8.786781359875599e-02, -1.757356271975120e-01, -8.786781359875599e-02, 1.912065771301972e+00, 3.824131542603945e+00, 1.912065771301972e+00, -6.103279191431599e+01, -1.220655838286320e+02, -6.103279191431599e+01, 1.621750246900412e+00, 3.243500493800824e+00, 1.621750246900412e+00, 1.717998769704532e+01, 3.435997539409063e+01, 1.717998769704532e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
