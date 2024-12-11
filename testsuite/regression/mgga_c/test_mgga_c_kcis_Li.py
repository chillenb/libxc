
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.342427552618962e-02, -8.371515710326172e-02, -4.959817863632871e-02, -1.808218740700230e-02, -1.095909861485380e-02, 2.098628406299194e-06, -1.643726357407228e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026962576935134e-01, -1.024963476191630e-01, -9.256356629933911e-02, -9.238935586969717e-02, -5.664197430173448e-02, -5.669262374811559e-02, -2.101480983705897e-02, -1.127518813766078e-01, -1.310471645140276e-02, -6.792553316104187e-02, -6.520150725925119e-06, 1.406421712804442e-05, -5.261616749615765e-13, -9.754474556692027e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.861769059769512e-04, 3.579307310313267e-04, 3.867950052208031e-04, 1.364079777225253e-03, 1.584861242149854e-03, 1.363229884931535e-03, 2.891917496006086e-01, 5.181750224872892e-01, 2.899937235023934e-01, 2.849725892383682e+01, 1.240351293287249e+01, 6.202321346314966e+00, 3.126378843899550e+02, 4.973769242805170e+02, 2.486950442972429e+02, 8.406092662569046e-04, 1.681208480457206e-03, 1.367054566944230e-03, 6.455592911137991e-09, 1.291118581827277e-08, 1.050300266018531e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.517920489624387e-05, -3.517051707674886e-43, -2.978197219499551e-42, -2.968540450469716e-42, -1.405766216307531e-38, -1.481935180641130e-38, -1.022608013947658e-32, -2.443810795953947e-06, -1.385656088160181e-31, -1.572871022279231e-08, -1.151971136269993e-09, -2.527468106101991e-06, -3.203490006160380e-19, -7.541133187092537e-21]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
