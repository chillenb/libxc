
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
    ref_tgt = numpy.asarray([-4.122787883119194e-02, -3.436429581908403e-02, -1.066995417488931e-02, -7.559315257803375e-04, -5.315291596642802e-09, -4.353983708784387e-05, -2.840259883800718e-08])
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
    ref_tgt = numpy.asarray([-8.448862380500032e-02, -8.430874180242982e-02, -7.339162750777838e-02, -7.322565183863956e-02, -2.966250119834828e-02, -2.949590553832442e-02, -1.565817335846574e-02, -1.079475873537298e-01, -4.185532834352115e-03, -4.460794300392486e-02, -1.724467028248359e-04, -1.531053798365528e-04, -9.717196170275458e-08, -1.278783889833075e-07])
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
    ref_tgt = numpy.asarray([1.157326624838861e-04, 6.147937556578381e-05, 1.156385477448291e-04, 2.664607412239756e-04, 2.026042608264738e-04, 2.663095224539777e-04, 6.326709115268325e-03, 1.042811294544413e-02, 6.351026967089999e-03, 1.900564268335213e+01, 4.862987484677894e+00, 2.432058622217670e+00, 2.442375509492325e+01, 2.591897797340521e+01, 1.296607114403985e+01, 5.210443896256941e-01, 1.042064888636250e+00, 5.215588946448408e-01, 1.321752999200642e+02, 2.643505997762938e+02, 1.321821336856628e+02])
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
    ref_tgt = numpy.asarray([-6.456170580497749e-03, -6.467444957563024e-03, -4.877636520417854e-03, -4.890877574680303e-03, -1.047322136034373e-03, -1.101491606789539e-03, -6.904924272081766e-01, -2.443810795953947e-06, -5.840882687653810e-02, -1.572871022272044e-08, -1.148406634714330e-09, -2.527468106101991e-06, -3.203183121790221e-19, -3.695772064639163e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
