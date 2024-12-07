
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_lak_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037415713190537e+00, -1.413085343588139e+00, -3.261087404612583e-01, -1.841019339613081e-01, -7.184056276152827e-02, -6.620982857770820e-03, -2.726417287219945e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_lak_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.691527987688159e+00, -2.694005225557767e+00, -1.851775918526557e+00, -1.893636405879868e+00, -2.271737172131767e-01, -4.634039553448760e-01, -2.443256446265737e-01, -1.124692927353208e-02, -8.692025583240440e-02, 9.515535964736274e-04, -1.186573177254171e-02, -1.184713626818029e-02, -5.924722364406827e-05, 4.207873899858663e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.209230563104357e-04, 0.000000000000000e+00, -1.204246592882878e-04, -5.200610058843631e-04, 0.000000000000000e+00, 4.561034488289250e-05, -2.999902391162552e-01, 0.000000000000000e+00, 1.227505756832476e-02, -1.867544142301667e+00, 0.000000000000000e+00, 2.805042349174488e+01, -6.376130737075395e+01, 0.000000000000000e+00, -2.127407550044933e+06, 2.493526739213772e+01, 0.000000000000000e+00, 2.519790213334071e+01, 4.130294837995839e+04, 0.000000000000000e+00, -8.773340776976423e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.651422677670951e-03, 6.641502912543069e-03, 9.826592392420357e-03, 9.791523465162211e-13, 7.522033489885034e-02, 3.722041384967294e-12, 7.441197863220247e-02, 8.990639119150012e-13, 1.697291823147360e-01, 8.776285445333061e-04, 2.204347260443516e-18, 9.141107136030131e-13, 3.399157810307923e-39, 3.865355492516790e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
