
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.996983960784007e+00, -1.384547403953022e+00, -3.792841441957842e-01, -1.800788210291084e-01, -7.440283338911975e-02, -2.053745905397894e-02, -3.838585904368181e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.550675343074331e+00, -2.553160244695431e+00, -1.745947013291060e+00, -1.747785338326829e+00, -3.432593228828136e-01, -3.444705055050146e-01, -2.328334549479369e-01, -2.609111414186117e-02, -7.785394591979253e-02, -8.296413305156971e-04, -2.742838671951816e-02, -2.723266425828722e-02, -5.541550226732571e-04, -3.939539840817354e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.116100269178062e-04, 0.000000000000000e+00, -5.098202893007814e-04, -1.506932966668228e-03, 0.000000000000000e+00, -1.500164313916166e-03, -8.427080715973698e-02, 0.000000000000000e+00, -8.345824921091832e-02, -9.737780388333960e+00, 0.000000000000000e+00, -4.352781648750259e-01, -5.948967393016184e+01, 0.000000000000000e+00, -2.785411237651393e+00, -4.423051982552122e-01, 0.000000000000000e+00, -4.130531457693814e-01, -2.027677897781292e+00, 0.000000000000000e+00, -2.902412778350464e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.670825042322612e-02, 2.668864381610370e-02, 2.730985377598797e-02, 2.728130498021741e-02, 2.079331917188827e-03, 2.038310665290869e-03, 3.604483701431960e-01, 1.181165623758641e-10, 4.886735340231405e-02, 6.325059422019450e-17, 3.822514513723540e-18, 1.354661166472653e-10, 1.445818868038081e-41, 1.053045407132662e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
