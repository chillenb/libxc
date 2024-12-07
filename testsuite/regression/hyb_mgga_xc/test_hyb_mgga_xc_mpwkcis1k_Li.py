
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpwkcis1k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.105092975254506e+00, -7.963680130802711e-01, -2.625871364119140e-01, -9.550400240376081e-02, -4.780425814596111e-02, -1.053664416655907e-03, -6.710685854965806e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpwkcis1k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.407329350028336e+00, -1.408399631931309e+00, -9.732729316964734e-01, -9.739086120330523e-01, -2.415050752146080e-01, -2.413291664101570e-01, -1.366283147511176e-01, -1.112419216450001e-01, -4.820858650888341e-02, -4.460850787001481e-02, -3.983041162039891e-03, -3.777418443466264e-03, -2.610947554162450e-07, -2.033982051678768e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.714481453028614e-05, 6.147937556578381e-05, -4.670173029715990e-05, -3.432151660908486e-04, 2.026042608264738e-04, -3.414638941058627e-04, -5.382098438002976e-02, 1.042811294544413e-02, -5.374461015040745e-02, 1.637700403084781e+01, 4.862987484677894e+00, 2.261655281596404e+01, -1.871935779885086e+01, 2.591897797340521e+01, 2.849504235361948e+02, 2.075912378009256e+01, 1.042064888636250e+00, 1.953558568661933e+01, 3.580614399886002e+02, 2.643505997762938e+02, 4.786635116117461e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwkcis1k_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwkcis1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.456170580497749e-03, -6.467444957563024e-03, -4.877636520417854e-03, -4.890877574680303e-03, -1.047322136034373e-03, -1.101491606789539e-03, -6.904924272081766e-01, -2.443810795953947e-06, -5.840882687653810e-02, -1.572871022272044e-08, -1.148406634714330e-09, -2.527468106101991e-06, -3.203183121790221e-19, -3.695772064639163e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
