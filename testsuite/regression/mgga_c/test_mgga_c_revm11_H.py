
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm11_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.274212843256261e-02, -4.254028377843427e-02, -5.770486922693769e-03, -1.028994622729177e-02, -3.304362960186103e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm11_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.817545766797384e-02, -5.341671942124414e+02, -8.388051964136782e-02, -1.399804587770806e+03, -2.632334661941064e-02, 4.714517603591162e+02, 3.861320525457059e-03, 1.492225173619677e+01, -4.204567563936985e-03, -1.487508768679539e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.718298996988338e-01, -9.436597993976675e-01, -4.718298996988338e-01, -1.924391263909280e-02, -3.848782527818560e-02, -1.924391263909280e-02, 4.774109380710734e-02, 9.548218761421468e-02, 4.774109380710734e-02, 3.854094045243655e-01, 7.708188090487346e-01, 3.854094045243655e-01, 5.938851365134210e-03, 1.187770271895288e-02, 5.938851365134210e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm11_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.157217494410808e+00, 1.156634342825354e+00, 9.585774657434455e-02, 9.521541890505623e-02, -1.319304235444578e-03, -1.316653036656145e-03, -2.283091367787895e-02, -2.283077692130066e-02, -7.780196734538369e-06, -7.780196779177671e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
