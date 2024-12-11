
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_m06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.030138794878935e+00, -1.309907982594728e+00, -1.349686563002160e-01, -1.821242650172076e-01, -4.527652279359657e-02, -9.115148941378802e-02, -1.552861059737350e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_m06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.238200131953525e+00, -3.239638993495509e+00, -1.788463295096476e+00, -1.789393013007106e+00, -3.623736787082820e-01, -3.702425997205074e-01, -1.721354832041090e-01, -1.147776647832923e-01, -8.757656307313426e-02, -3.690797565161724e-03, -1.223393198269482e-01, -1.197162972091348e-01, -2.465327255215663e-03, -9.714983726534188e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.840261500914240e-04, 0.000000000000000e+00, -3.827841963449345e-04, -1.639980229844878e-03, 0.000000000000000e+00, -1.634055872916901e-03, -7.433889507808795e-01, 0.000000000000000e+00, -7.392728047698728e-01, -5.589736185667308e+00, 0.000000000000000e+00, 3.689483856493438e-01, -2.838687762606033e+02, 0.000000000000000e+00, 2.382516619603516e+00, 1.586712046334569e-04, 0.000000000000000e+00, 3.499139806458657e-01, 1.087494783104993e-10, 0.000000000000000e+00, -1.558006159987078e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.583303787564868e-02, 7.556739411332762e-02, 6.243174869358658e-03, 6.154375242591008e-03, 3.346150654244723e-02, 3.511918663359638e-02, -2.012488522129150e+00, -1.803586846019249e-04, 2.278105336318756e-01, -3.696306054933462e-08, -8.973833717667498e-08, -1.946581160160833e-04, -5.029631185970785e-19, -6.136293873515274e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
