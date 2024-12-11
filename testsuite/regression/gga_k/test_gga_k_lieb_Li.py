
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lieb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lieb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.665015103925978e+01, 8.409957017239657e+00, 1.038640101114823e+00, 1.335414009487452e-01, 3.369908369591745e-02, 5.741152769360279e-01, 2.522599351623472e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lieb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lieb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.563559972463561e+01, 2.568324802604453e+01, 1.197338995500457e+01, 1.199467537341618e+01, 1.380279931827693e-01, 1.356206655744189e-01, 2.123744630728761e-01, -5.672989658792861e-01, 2.063949645927302e-02, -2.249913535764627e-01, -5.625568374350544e-01, -5.822438816017877e-01, -2.637018849519042e-01, -2.204112080468699e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lieb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lieb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.582829330040256e-03, 0.000000000000000e+00, 3.573162078721730e-03, 1.070528665069806e-02, 0.000000000000000e+00, 1.067799793633625e-02, 7.721564860237757e-01, 0.000000000000000e+00, 7.731424535003864e-01, 4.842994836644718e+00, 0.000000000000000e+00, 1.455633934990636e+04, 7.773825926073611e+01, 0.000000000000000e+00, 4.562911917586081e+08, 1.251814844410095e+04, 0.000000000000000e+00, 1.279530671565867e+04, 1.531174703469454e+09, 0.000000000000000e+00, 4.261704909931756e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
