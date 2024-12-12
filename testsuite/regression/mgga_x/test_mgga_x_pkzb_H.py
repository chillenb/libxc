
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_pkzb_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.308291579245066e-01, -5.804740242324986e-01, -3.584342879289485e-01, -1.448654896092220e-01, -7.399213127025143e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_pkzb_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.402367387732126e-01, 8.893237857901796e-18, -7.281829113410455e-01, -1.422409802618665e-16, -3.924510295471247e-01, -3.532878145637116e-17, -1.376115295351458e-01, -4.727102753500217e-17, -9.865598626982247e-03, 2.838590470703411e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.558838611927446e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.524701184427820e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.268519773839489e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.264812256239624e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.054875148780281e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.631916989449985e-03, 0.000000000000000e+00, 1.095262457535168e-02, 0.000000000000000e+00, 2.213296580046790e-02, 0.000000000000000e+00, 2.110342910823889e-02, 0.000000000000000e+00, 5.890216024174090e-09, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
