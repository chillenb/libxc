
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_br89_explicit_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.382820101341553e+00, -1.207208207798621e+00, -4.831878671393649e-01, -1.469658346993374e-01, -8.663521678566091e-02, -1.709566510076637e-01, -4.054693415659630e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_br89_explicit_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.870071603816722e+00, -1.871844409525223e+00, -1.735703912676059e+00, -1.737311602927852e+00, -4.074251154139101e-01, -4.073045826266704e-01, -2.236308453775429e-01, -6.431252498914869e-02, -8.184206227219791e-02, -1.224122206783735e-02, -6.402030496755348e-02, -6.357104123727778e-02, -9.170919848053289e-03, -6.568017869295429e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.727402101351804e-07, 0.000000000000000e+00, -7.727216460208858e-07, -1.235395499185506e-03, 0.000000000000000e+00, -1.232052749165679e-03, -9.978002476703959e-02, 0.000000000000000e+00, -9.975297625019068e-02, -2.576313885587899e+00, 0.000000000000000e+00, -1.056917538454941e+03, -7.978128406606447e+01, 0.000000000000000e+00, -3.891160144362611e+07, -9.251822794315294e+02, 0.000000000000000e+00, -9.360131093815061e+02, -1.158576410190451e+08, 0.000000000000000e+00, -3.469082344291608e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_1_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.002416624446001e-05, -1.005104532868840e-05, -5.363503690105370e-03, -5.362660941506672e-03, -6.005914609978812e-03, -5.996629414978154e-03, -2.472439298343039e-02, -3.374658281257540e-03, -4.769875901104283e-02, -3.963491115823231e-03, -3.435010574141752e-03, -3.399946632728157e-03, -3.516744409409053e-03, -3.783311525374734e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.009666497784006e-05, 4.020418131475361e-05, 2.145401476042149e-02, 2.145064376602675e-02, 2.402365843991525e-02, 2.398651765991262e-02, 9.889757193372156e-02, 1.349863312503016e-02, 1.907950360441714e-01, 1.585396446329292e-02, 1.374004229656701e-02, 1.359978653091260e-02, 1.406697763763621e-02, 1.513324610149894e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
