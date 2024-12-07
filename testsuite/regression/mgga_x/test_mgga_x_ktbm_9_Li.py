
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_9_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.083894588632850e+00, -1.456487083737452e+00, -2.802409975225909e-01, -1.868562650227405e-01, -6.539047809526782e-02, -9.826488083325091e-03, -1.795840341338003e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_9_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.603421137343709e+00, -2.605882290403919e+00, -1.768320677584882e+00, -1.769663472661344e+00, -3.522818352401342e-01, -3.538533435994957e-01, -2.394270623113973e-01, -1.167764670132977e-02, -8.510191107675512e-02, -3.703035481531691e-04, -1.267395803542974e-02, -1.219064446190796e-02, -2.553159697670731e-04, -1.758371558405988e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_9_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.912774688368158e-04, 0.000000000000000e+00, -5.892441690213631e-04, -2.368817779533899e-03, 0.000000000000000e+00, -2.362226391315145e-03, -7.313130884894152e-02, 0.000000000000000e+00, -7.678414624885803e-02, -8.968639014066731e+00, 0.000000000000000e+00, -3.309680055488921e+01, -9.319703662130065e+01, 0.000000000000000e+00, -8.300870813349209e+04, -4.075686774597059e-02, 0.000000000000000e+00, -2.958995884758614e+01, -5.650326864837385e-02, 0.000000000000000e+00, -3.757943855078613e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_9_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_9_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_9", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.434378321786049e-02, 2.431262191581396e-02, 4.048670512735573e-02, 4.045072895118784e-02, 2.375602711630292e-02, 2.559629041459275e-02, 2.457666214270071e-01, 4.230828302786042e-04, 3.391444563054426e-01, 3.382076390508055e-05, 1.228256282839694e-08, 4.303375912078528e-04, 5.432256630133685e-17, 1.639335727991930e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
