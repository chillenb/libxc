
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pbe1kcis_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.568686177483735e-01, -4.659336392608358e-01, -2.794703835425853e-01, -8.118071600404225e-02, -4.769008711327520e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pbe1kcis_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.178743135329482e-01, -2.236903732303361e-01, -6.147585789763784e-01, -2.146767187768155e-01, -3.677690438577416e-01, -1.713347950604392e-01, -1.062609883146090e-01, -8.486840964027169e-02, -6.267287109763178e-03, -7.156556982179594e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.297132785570406e+00, 4.142665981990480e-02, 8.608116618147828e+13, 5.568065621082403e-02, 7.485113257561367e-02, 8.596311204654978e+13, 3.009688764192422e-01, 6.566317903568427e-01, 8.612673461976202e+13, 3.538678202281189e+01, 1.312306249311276e+02, 7.269160365222498e+13, 5.691075487228878e+12, 1.138216485897558e+13, 7.838286254346245e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.114511467427907e+00, -3.995554316157371e-04, -1.776359381880728e-38, -3.990074693494971e-04, -3.171872743719510e-37, -3.997669429258967e-04, -4.440135579057598e-34, -6.330126426738284e+00, -2.646759132179996e-27, -6.330142115625751e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
