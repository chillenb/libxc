
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpwkcis1k_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.234164536008855e-01, -3.600322519192958e-01, -2.175290838741772e-01, -6.463883387793430e-02, -3.989712133079993e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpwkcis1k_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.802938159647467e-01, -2.236903732303360e-01, -4.735567291876577e-01, -2.146767187768155e-01, -2.851806442998622e-01, -1.713347950604392e-01, -8.420514547977145e-02, -8.486840964027167e-02, -5.228224993299837e-03, -7.156556982179594e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.299388592773095e+00, 4.142665981990480e-02, 8.608116618147828e+13, 6.738298167437357e-02, 7.485113257561367e-02, 8.596311204654978e+13, 4.009590946546272e-01, 6.566317903568427e-01, 8.612673461976202e+13, 5.504477082724252e+01, 1.312306249311276e+02, 7.269160365222498e+13, 5.691079478099723e+12, 1.138216485897558e+13, 7.838286254346245e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.114511467427907e+00, -3.995554316157371e-04, -1.776359381880728e-38, -3.990074693494971e-04, -3.171872743719510e-37, -3.997669429258967e-04, -4.440135579057598e-34, -6.330126426738284e+00, -2.646759132179996e-27, -6.330142115625751e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
