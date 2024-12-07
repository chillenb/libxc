
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_mol_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.196093851580521e-02, -1.624163927549032e-02, -6.499757552312108e-03, -1.035552641082561e-04, -1.475794651414897e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_mol_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.710774619508801e-02, 2.080776106020670e+00, -4.017648383066519e-02, 8.418032512748297e+01, -2.432341329842166e-02, 3.984234972459666e+01, -6.237386522360017e-04, 2.255552756171166e-01, -9.602072806574186e-10, 6.759073257036133e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_mol_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.052076786439451e-02, 4.104153572878902e-02, 2.052076786439451e-02, 1.077081421263397e-02, 2.154162842526794e-02, 1.077081421263397e-02, 3.865672166627838e-02, 7.731344333255680e-02, 3.865672166627838e-02, 5.849386882116357e-02, 1.169877376423272e-01, 5.849386882116357e-02, 6.288552985890233e-04, 1.257710595914521e-03, 6.288552985890233e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
