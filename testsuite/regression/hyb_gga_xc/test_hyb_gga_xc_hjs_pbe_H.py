
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_pbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.135601235318829e-01, -4.665285482719893e-01, -2.928424223609762e-01, -1.153741247798607e-01, -7.393367408852051e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_pbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.733186294123125e-01, 1.596983044605467e+00, -5.950779270812017e-01, 7.387469275550377e+01, -3.434186914841975e-01, 4.139501778593020e+01, -1.191089647119090e-01, 3.382974671448072e-01, -9.849573520910686e-03, 1.066995416126142e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_pbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.414018626287509e-03, 3.288056552649452e-02, 1.644028276324726e-02, -7.887613577946227e-03, 2.041382516785488e-02, 1.020691258392744e-02, -8.673837565723372e-02, 8.417394271524091e-02, 4.208697135762044e-02, -3.437015441655412e+00, 1.762212799391784e-01, 8.811063996958873e-02, -5.507290719455485e+00, 1.985451945004567e-03, 9.927259729551347e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
