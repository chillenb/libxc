
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_mohlyp2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.863691212690667e+00, -1.318602016803910e+00, -5.146505953667477e-01, -1.653315313667248e-01, -8.880823270603075e-02, -3.562738083664197e-02, -6.617419943528887e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_mohlyp2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.426254490846706e+00, -2.428455852736546e+00, -1.630642505732620e+00, -1.632093908914943e+00, -3.008692749501607e-01, -3.015651869270463e-01, -2.187756385154190e-01, -8.502660622400991e-02, -4.792296560075483e-02, -1.654750465181900e-02, -4.750098211314298e-02, -4.721971227469067e-02, -9.440355403565742e-04, -7.105358081520985e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_mohlyp2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.683649974858365e-05, 2.611407710925856e-06, -8.643775719206363e-05, -6.440859267551988e-04, 1.823470894624294e-05, -6.416243129512329e-04, -1.993501719353557e-01, 2.386881431793093e-02, -1.989388263198622e-01, -8.408866063254908e-01, 2.298067384726520e+00, 6.962775399967789e-01, -1.542442221934402e+02, 1.178469867164830e+01, 2.260791650686114e+00, -1.024191649990045e+00, 3.968048660888829e-02, -9.549730375543827e-01, -4.788349665965280e+00, 0.000000000000000e+00, -6.854030712790340e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
