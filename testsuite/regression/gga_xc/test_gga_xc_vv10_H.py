
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_vv10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.539243552393891e-01, -5.985564556060090e-01, -3.791880093913159e-01, -1.390328202817085e-01, -1.969758192014154e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_vv10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.654752089202350e-01, 1.596983044605467e+00, -7.390282599447897e-01, 7.387469275550377e+01, -4.184738693926705e-01, 4.139501778593020e+01, -1.354591750083135e-01, 3.382974671448072e-01, -1.583648959489277e-02, 1.066995416380574e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_vv10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.469036770165913e-03, 3.288056552649452e-02, 1.644028276324726e-02, -2.507433002986339e-02, 2.041382516785488e-02, 1.020691258392744e-02, -1.848716341438741e-01, 8.417394271524091e-02, 4.208697135762044e-02, -5.612933283718303e+00, 1.762212799391784e-01, 8.811063996958873e-02, -8.331951337412362e+03, 1.985451945004567e-03, 9.927259729551347e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
