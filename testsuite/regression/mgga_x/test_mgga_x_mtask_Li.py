
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mtask_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.042462005003014e+00, -1.337193027796430e+00, -9.969129603860488e-02, -1.895534906925928e-01, -3.665136297108270e-02, -5.292504163495639e-03, -7.198921264329634e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mtask_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.933806171752156e+00, -2.936654179188120e+00, -2.037800251410063e+00, -2.039423518068323e+00, -2.659773994916101e-01, -2.702716626399830e-01, -2.677384411419577e-01, -1.235821017466763e-02, -8.668705294420696e-02, -1.215023688407644e-04, -6.495216701200453e-03, -1.301772211779925e-02, -3.718780254271426e-06, 3.275998989990509e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mtask_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.577478003151615e-04, 0.000000000000000e+00, -5.559779256741763e-04, -1.989787064112582e-03, 0.000000000000000e+00, -1.983452305589982e-03, 1.576350171378058e-01, 0.000000000000000e+00, 1.521598901978843e-01, -1.079852086214289e+01, 0.000000000000000e+00, 3.082201543917023e+01, -4.280644352645353e+01, 0.000000000000000e+00, 2.925418727136550e+04, 3.366788854826642e-01, 0.000000000000000e+00, 2.768764359671449e+01, 2.132779474140229e-02, 0.000000000000000e+00, 3.474231786794352e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mtask_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.018707886163724e-02, 3.017040944966270e-02, 3.740899923864710e-02, 3.737958227775741e-02, 2.440572990602879e-02, 2.534846974204675e-02, 4.244485202980049e-01, 1.669945474956759e-12, 3.166536302630667e-01, 1.937308826464924e-16, 8.285542409876060e-16, 1.776176721797819e-12, 9.714745002320253e-33, 5.550957003238005e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
