
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.968910313143120e+00, -1.367175144599829e+00, -3.999809100268318e-01, -1.768456266327649e-01, -7.665000786334736e-02, -2.054447947414327e-02, -3.838586978645408e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.407095380125136e+00, -2.409267794785516e+00, -1.706387075607042e+00, -1.708133825453897e+00, -3.508899280792214e-01, -3.515788134232583e-01, -2.137875858320913e-01, -2.611569207866658e-02, -7.454096060771728e-02, -8.296433204579698e-04, -2.745718545520368e-02, -2.725989618423128e-02, -5.541555286317482e-04, -3.939542015705412e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.902250936707573e-04, 0.000000000000000e+00, -9.893677216840785e-04, -1.582068021477130e-03, 0.000000000000000e+00, -1.579292200849844e-03, -8.984262387141041e-02, 0.000000000000000e+00, -8.918912355359328e-02, -2.805571268378494e+01, 0.000000000000000e+00, -2.780506880868454e-01, -6.386060361197524e+01, 0.000000000000000e+00, -1.776498264494791e+00, -2.825911407939073e-01, 0.000000000000000e+00, -2.638791781080498e-01, -1.293222808933014e+00, 0.000000000000000e+00, -1.851114589210472e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.133256656990907e-02, 5.146081972621280e-02, 2.699536652654771e-02, 2.707549740077117e-02, 5.249170907102530e-04, 4.408895734294224e-04, 1.012299727590791e+00, 6.714990221518002e-11, 1.276521894098830e-02, 3.628503406176095e-17, 4.835635833592273e-17, 7.697046751037394e-11, 7.924103401730547e-38, 6.041099655890685e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
