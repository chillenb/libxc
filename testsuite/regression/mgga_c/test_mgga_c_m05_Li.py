
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.352896657886006e-02, -8.371482262002408e-02, -4.959806172627838e-02, -1.808781036521564e-02, -1.095911493760681e-02, -3.629454026131921e-03, -8.776912343101520e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.023869984579563e-01, -1.024892757097885e-01, -9.254539378426756e-02, -9.240668603323737e-02, -5.664537600732748e-02, -5.668890672673783e-02, -2.101683837828089e-02, -1.144186445709685e-01, -1.310473996224768e-02, -7.929786192594963e-02, -5.271410379246663e-03, -5.413772983921572e-03, -7.067557114754856e-05, -3.102176349053440e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.920312695751397e-04, 0.000000000000000e+00, -2.973221672102022e-04, -2.240540120390880e-03, 0.000000000000000e+00, -2.229435175535608e-03, -2.037897171097606e+00, 0.000000000000000e+00, -2.041419810109298e+00, -7.641589714617592e+00, 0.000000000000000e+00, -1.213790986465825e+02, -1.792587518005176e+03, 0.000000000000000e+00, 5.406861142509741e+04, 3.185231368464900e-01, 0.000000000000000e+00, 1.538472838214645e+01, 1.092533063955760e+00, 0.000000000000000e+00, -1.144420862261128e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.526346979962778e-05, -7.577336033803976e-46, -6.416391505146912e-45, -6.395586446153691e-45, -3.028647150211534e-41, -3.192749052890911e-41, -2.203149546907371e-35, -2.222554551342977e-04, -2.985309460798631e-34, -2.874695678759458e-05, -4.773084468328686e-06, -2.235763763059440e-04, -1.326510540137745e-10, -2.553864405538792e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
