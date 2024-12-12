
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.473388561422263e-01, -4.905425768635358e-01, -2.861664324098356e-01, -7.274733080839879e-02, -1.628172305293030e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.296160398659752e-01, -2.199798770977095e-01, -6.463515033481760e-01, -2.101325999806031e-01, -3.726351240772403e-01, -1.813309671284328e-01, -8.431071379901590e-02, -1.091393373285240e-01, -1.090302779333202e-02, -9.891645420874053e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.916933699869419e-03, 8.472540629547595e-03, 4.236270314773800e-03, -1.044278195182385e-02, 9.771615395811520e-03, 4.885807697905760e-03, -6.433788775764307e-02, 9.485995032994637e-02, 4.742997516497319e-02, -5.263260295688980e+00, 2.306501582130026e+01, 1.153250791065013e+01, 1.755108129159076e+04, 4.144222864515410e+05, 2.072111432257705e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.649697363287765e-02, -9.984991666357742e-03, 1.888127277049634e-02, -8.263185613481468e-03, 2.469076704103378e-02, -1.625762109617745e-02, 4.020910121946289e-02, -7.556336542252602e-02, -1.159034326232364e-02, -1.419598229148764e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
