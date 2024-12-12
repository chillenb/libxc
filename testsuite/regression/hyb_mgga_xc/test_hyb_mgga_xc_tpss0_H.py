
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpss0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.282065416703160e-01, -4.740912370458291e-01, -2.779715645815429e-01, -9.858829209444767e-02, -5.547461018925360e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpss0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.861516344637013e-01, -6.412806689281922e-02, -6.175928002461044e-01, -3.635674312071321e-02, -3.637165480253491e-01, -1.617448765558878e-02, -9.434833689372316e-02, -3.172051243992244e-04, -7.391414925447553e-03, -4.663822954112051e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.392547494337496e-01, 5.246030108347647e+00, 2.623015054173823e+00, -1.799692221646532e-02, 9.006775425259457e-02, 4.503387712629727e-02, -1.443402706345925e-02, 1.883981439556215e-01, 9.419907197781079e-02, -4.140703588414610e+00, 1.904245066374393e-01, 3.295388761723378e+00, -4.154499233624186e+00, 1.987619836265168e-03, 1.036202013725005e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.760156674157844e+00, -6.182509960461265e+00, 3.094285643695056e-02, -7.616412850374940e-02, -3.642599047454440e-04, -3.228871222511654e-02, -3.729796988339954e-04, -6.238502800849651e-04, -6.466867300223765e-10, -6.808566129057391e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
