
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_pkzb_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.935297838124740e-12, -3.118079983577066e-02, -2.518518753751072e-02, -1.327196220086343e-02, -1.569796946832078e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_pkzb_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.412739203652180e-02, 5.729811765330128e+00, -3.508811108100187e-02, -2.505892439941811e-01, -2.872211520391200e-02, -1.951032357820481e-01, -1.571598845536356e-02, -9.049038861325961e-02, -2.001666154573393e-03, -7.028460031535858e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.623015059775090e+00, 5.030726432403695e-02, -4.143300324422852e+12, 2.611078413273257e-02, 5.222156826546515e-02, 2.611078410924460e-02, 2.231029244572280e-01, 4.462058489144561e-01, 2.231029243733294e-01, 4.386184838832986e+01, 8.772369677665974e+01, 4.386184755781915e+01, 8.905587071022807e+06, 1.781117414204561e+07, 8.905587070505178e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_pkzb_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.255850533922394e+00, 3.314640259538302e+00, 5.014341661433584e-45, 5.015323694811061e-45, 8.421436650632151e-44, 8.422009016238110e-44, 1.595430357772155e-35, 1.595430693164418e-35, 7.059212617507213e-29, 7.059212618423918e-29]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
