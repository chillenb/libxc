
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cs_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.337346591735538e-16, -5.777800109910487e-16, -2.290254649861370e-15, -5.077404922397761e-14, -2.865040878004794e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cs_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.746903433430564e-17, -1.292146032167648e-01, -1.981884743293375e-16, -1.242980638337787e-01, -1.583420752264289e-15, -9.854944047043189e-02, -6.529168379846032e-14, -4.158882017096325e-02, -9.431025128624343e-12, -2.453540189748325e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cs_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.363307775330318e-17, 1.472661555066064e-16, 7.363307775330318e-17, 1.620034967815492e-16, 3.240069935630985e-16, 1.620034967815492e-16, 6.948838539038938e-15, 1.389767707807788e-14, 6.948838539038938e-15, 1.559146049170965e-11, 3.118292098341930e-11, 1.559146049170965e-11, 6.268688879976413e-26, 1.253737775995283e-25, 6.268688879976413e-26])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cs_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-7.522451842558554e-32, -2.169436830976663e-17, -1.621402329392308e-31, -3.424881429235486e-17, -6.903506267427218e-30, -2.977325616464631e-16, -1.559159341234023e-26, -1.276980726713362e-14, -6.268688893000043e-41, -5.368328357205251e-33])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cs_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.756137568036512e-16, -5.780540928528317e-31, -2.785308814865537e-16, -1.262390896823658e-30, -2.390546400271758e-15, -5.526788394415427e-29, -1.021611046584717e-13, -1.247265140208141e-25, -4.294662652280686e-32, -5.014951166539975e-40])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
