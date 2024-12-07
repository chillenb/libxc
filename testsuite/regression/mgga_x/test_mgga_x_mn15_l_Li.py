
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mn15_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.590992868896378e+00, -1.481654626222060e+00, -2.827994244194554e-01, -1.356774142039510e-01, -1.061130171404844e-01, -2.306741165848906e-02, -4.857628525217893e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mn15_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.229047608316201e-02, -3.680841787337789e-02, -1.230384725992684e+00, -1.227794902309111e+00, -4.996155155512091e-01, -5.089790868530770e-01, -2.896340529948662e-01, -2.861005258946359e-02, -5.906356173251544e-02, -1.047497795020177e-03, -2.977159529691870e-02, -2.967065531763801e-02, -7.007366791052589e-04, -4.986025257593731e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn15_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.499941884155908e-03, 0.000000000000000e+00, -1.495520302796844e-03, -4.548210954427082e-03, 0.000000000000000e+00, -4.540758675190077e-03, -1.033428609693197e-01, 0.000000000000000e+00, -1.018755681825502e-01, 3.348887414866550e+01, 0.000000000000000e+00, -1.060669746726801e+00, -2.865017309280393e+02, 0.000000000000000e+00, -7.304543768569769e+00, -1.077892332305172e+00, 0.000000000000000e+00, -1.003139506221568e+00, -5.321577499502895e+00, 0.000000000000000e+00, -7.620680991065874e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn15_l_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn15_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.515266765958194e-01, -1.521217313199266e-01, 1.373367846737920e-02, 1.336261462419571e-02, 6.799436022271504e-02, 6.962494833502675e-02, 4.960863605990357e-01, 7.314209482050236e-06, 4.537312938755618e-01, 1.450354763208550e-09, 3.634522378187755e-09, 7.905968280815082e-06, 1.971973087588105e-20, 1.617335626145502e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
