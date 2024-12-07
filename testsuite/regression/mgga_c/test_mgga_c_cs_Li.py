
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.488620635314217e-01, -2.830882704387487e-02, -1.252349759847314e-02, -2.138845533350452e-05, -5.388673489777090e-09, -2.091265111694189e-03, -3.003770849422545e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.643974847173764e-01, -1.657187908130363e-01, -6.383846385133332e-02, -6.372567877843027e-02, -1.096588442130798e-01, -1.091798872233294e-01, -5.617601913887680e-06, -6.428692050610983e-02, -2.023530745967225e-09, -3.162926993457284e-02, -2.753034837986895e-03, -2.838357579094408e-03, -2.079686215637140e-05, -9.360839371319271e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.767598563812830e-05, 3.535197127625659e-05, 1.767598563812830e-05, 9.887202714802811e-05, 1.977440542960562e-04, 9.887202714802811e-05, 4.869522013062116e-02, 9.739044026124233e-02, 4.869522013062116e-02, 1.147121369241314e-03, 2.294242738482627e-03, 1.147121369241314e-03, 3.011635510143268e-06, 6.023271020286536e-06, 3.011635510143268e-06, 5.824276345979157e-03, 1.164855269195831e-02, 5.824276345979157e-03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cs_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.149586877712960e-04, -1.146485042756393e-04, -2.151763220176123e-04, -2.146278187051317e-04, -1.463651513874309e-03, -1.465520452121790e-03, -1.831336167424412e-09, -5.504352495917766e-06, -1.533808701622597e-16, -9.002818016341541e-10, -1.057796549648482e-08, -1.081216711557365e-08, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-9.171880342051149e-04, -9.196695021703675e-04, -1.717022549641054e-03, -1.721410576140899e-03, -1.172416361697433e-02, -1.170921211099447e-02, -4.403481996735186e-05, -1.465068933939866e-08, -7.202254418891984e-09, -1.227046960831373e-15, -8.649733692458917e-08, -8.462372397187855e-08, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
