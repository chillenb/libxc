
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pbe1kcis_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.447650294103858e+00, -1.023897742833296e+00, -2.753160146114501e-01, -1.404673796440289e-01, -5.948961803201722e-02, -1.603226017947243e-02, -2.722610604882991e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pbe1kcis_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.907383187794294e+00, -1.908898524051290e+00, -1.345606407590785e+00, -1.346498715465941e+00, -3.576630559743925e-01, -3.575856896628645e-01, -1.842425687623292e-01, -1.331559995073160e-01, -7.781208658821953e-02, -6.857265768403518e-02, -2.146283766804187e-02, -2.128619978221872e-02, -4.322420077509029e-04, -1.703351317555493e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.694596595111207e-04, 3.579307310313267e-04, 1.708305517651723e-04, 4.313058737837814e-04, 1.584861242149854e-03, 4.336249304586890e-04, 9.130889660456831e-03, 5.181750224872892e-01, 9.455948751985355e-03, 2.525786327677401e+01, 1.240351293287249e+01, 6.202321346314966e+00, 1.814707386159646e+02, 4.973769242805170e+02, 2.486950442972429e+02, 8.406092662569046e-04, 1.681208480457206e-03, 1.367054566944230e-03, 6.455592911137991e-09, 1.291118581827277e-08, -2.731636994798592e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pbe1kcis_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pbe1kcis", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.517920489624387e-05, -3.517051707674886e-43, -2.978197219499551e-42, -2.968540450469716e-42, -1.405766216307531e-38, -1.481935180641130e-38, -1.022608013947658e-32, -2.443810795953947e-06, -1.385656088160181e-31, -1.572871022279231e-08, -1.151971136269993e-09, -2.527468106101991e-06, -3.203490006160380e-19, -7.541133187092537e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
