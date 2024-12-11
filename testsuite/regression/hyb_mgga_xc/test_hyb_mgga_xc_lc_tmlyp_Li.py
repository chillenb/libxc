
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_lc_tmlyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.781599704688474e+00, -1.168749812528588e+00, -7.870002542318121e-02, 1.863260106231263e+164, 2.017918590621531e+166, 2.623353823650982e+218, -1.050047488198356e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_lc_tmlyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.494461274472217e+00, -2.496750221351299e+00, -1.709404411344364e+00, -1.710771171733891e+00, -3.152541374225760e-01, -3.151052640679465e-01, -1.753360213355373e+164, -4.337504611384996e+164, -2.505540162817595e+165, -1.918491662790991e+166, 9.985391048778975e+218, 1.006762709573265e+219, 1.817649879371118e-05, -9.360849199043402e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.704412155672172e-04, 5.222815421851711e-06, -2.694840655367811e-04, -1.328692915519192e-03, 3.646941789248587e-05, -1.323762271949794e-03, -8.957878453908331e-01, 4.773762863586187e-02, -8.769372437805830e-01, -4.022588657532278e-01, 4.596134769453040e+00, -2.735152374873195e+00, -1.222374128540228e+00, 2.356939734329661e+01, 1.006662589479425e+01, -3.037603040281809e+00, 7.936097321777658e-02, -6.063633201872509e+00, -6.477104750102472e+00, 0.000000000000000e+00, -1.387201669422343e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_lc_tmlyp_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_lc_tmlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.209368006626644e-02, 1.208717184103747e-02, 1.701291832849439e-02, 1.699974819263177e-02, 4.048982096292891e-02, 4.048972663651196e-02, 2.864618489939052e-02, 2.486855527300091e-05, 4.479314341562526e-03, 9.452377746616706e-10, 1.570309856668946e-05, 2.798287213837665e-05, 2.450987886140704e-10, 1.014076897073678e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
