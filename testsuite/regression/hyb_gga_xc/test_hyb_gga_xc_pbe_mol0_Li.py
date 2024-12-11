
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe_mol0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_mol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.411658346908322e+00, -1.014796230608229e+00, -3.239120970778214e-01, -1.351070822863825e-01, -6.337765457810952e-02, -1.541026456097032e-02, -2.878940523657158e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe_mol0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_mol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.784596656705687e+00, -1.786082161858257e+00, -1.225231045064213e+00, -1.226165364791647e+00, -3.287796460465872e-01, -3.289483924172891e-01, -1.780125821021235e-01, -1.103218626470099e-01, -6.380074586899442e-02, 2.656993960297310e-01, -2.060073560742555e-02, -2.045234148255463e-02, -4.156167829033139e-04, -2.954657098232306e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe_mol0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_mol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.865308974711057e-04, 9.770865834560218e-05, -1.857288214517279e-04, -7.612442862657272e-04, 3.055252754263828e-04, -7.583514499604478e-04, -5.191663878873413e-02, 4.548929910098821e-03, -5.175976490458437e-02, 3.472240385841752e-01, 8.046127959984386e+00, 3.857217666560185e+00, -4.469906436342254e+01, 1.725728316564792e+01, 7.568318744313009e+00, -1.684386296619522e-01, 2.127118198217388e-04, -1.572806953465715e-01, -7.718743004667005e-01, 2.035444796919866e-06, -1.104858872120255e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
