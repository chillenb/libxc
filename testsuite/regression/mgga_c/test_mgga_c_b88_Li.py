
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.533582005867859e-02, -3.842476348058225e-02, -1.057882930384721e-02, -1.748609739533901e-04, -4.406274436762527e-07, -8.571935356279999e-01, -9.261095740894912e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.473911289192630e-02, -6.457925236495018e-02, -6.664499338482577e-02, -6.648806351212580e-02, -4.015392035821734e-02, -3.955309528253163e-02, -1.408937500186742e-03, -3.852168530268050e-01, -5.551796272753165e-03, -3.301091205969034e+00, -2.027157968395090e+00, -3.624589378297419e-02, -2.017057159399027e+03, -8.168472270362201e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.284206582472372e-05, 0.000000000000000e+00, 4.259252674858509e-05, 2.044812562301521e-04, 0.000000000000000e+00, 2.033554803661422e-04, 2.756410840872834e-02, 0.000000000000000e+00, 2.710616475849925e-02, 1.732434230109489e+00, 0.000000000000000e+00, 3.001705175415247e+03, 3.239377555291724e+01, 0.000000000000000e+00, 1.846071213420488e+09, 2.287263297174329e+04, 0.000000000000000e+00, 7.884531652321249e+02, 5.326208502407798e+12, 0.000000000000000e+00, 1.579199298948784e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.192563645025942e-03, -1.190931263182908e-03, -1.991196048135936e-03, -1.988805073980728e-03, -5.290526730111915e-03, -5.279999621385669e-03, -6.615017724529870e-02, -1.161169913169434e-02, -7.746886454364355e-02, -9.334849117717287e-03, -1.167309360527918e-02, -1.143978317059454e-02, -7.027468971594636e-03, -6.888790814604931e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
