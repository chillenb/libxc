
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk_loc4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.217380335279993e+00, 4.411920008317717e+00, 3.416710351683373e+00, 2.577498149504321e-02, 7.226988855177095e-02, 6.790618985826407e+01, 1.144554864212650e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk_loc4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.217356802578061e+00, -4.217263685220781e+00, -1.041317624949138e+00, -1.029240527950765e+00, 1.264789177531118e+00, 1.205530574756432e+00, 1.678206436167268e-02, 5.054534597553484e-01, -5.990713140812928e-02, -1.210226358618995e+00, 5.014518844872965e-01, 5.188933394301501e-01, 2.340442352496197e-01, -1.185586352303033e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.927193037561568e-02, 0.000000000000000e+00, 1.921993050708768e-02, 4.963361324999619e-02, 0.000000000000000e+00, 4.947682039302235e-02, -6.853121063186742e-01, 0.000000000000000e+00, -6.167412989765378e-01, 2.324752318952768e+01, 0.000000000000000e+00, -1.291918909342438e+04, 3.784109146366873e+02, 0.000000000000000e+00, 2.454376727176486e+09, -1.111023334654098e+04, 0.000000000000000e+00, -1.135621965072012e+04, -1.358963613974630e+09, 0.000000000000000e+00, 2.292358375079883e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc4_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([1.794124233207086e-08, 1.810305285490883e-08, 2.573018610102923e-02, 2.582841564820683e-02, 2.171249997765634e-01, 2.140127693011994e-01, 2.005225148158088e-02, 2.171249999999520e-01, 1.771280259892346e-02, 0.000000000000000e+00, 2.171250000000001e-01, 2.171250000000002e-01, 2.171249999999998e-01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
