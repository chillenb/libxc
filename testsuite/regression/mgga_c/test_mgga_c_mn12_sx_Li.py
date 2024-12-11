
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.236520272154800e-02, -4.553569612073346e-02, -5.903875755703740e-01, -1.780971771938535e-02, -1.057116672953833e-01, -1.401226086732070e-01, -3.476903582622099e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.217186691443571e-01, -1.215587865814881e-01, -1.294236618327339e-01, -1.293482133780538e-01, -4.624280989012244e-02, -4.676097521757783e-02, -2.591048342083157e-02, -1.276271269154960e-01, 2.556692533473513e-02, -5.379785671961739e-01, -1.761161966739689e-01, -1.780925990524860e-01, -4.090219255075959e-03, -6.001823714128680e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.361286376163857e-04, -8.722572752327713e-04, -4.361286376163857e-04, -9.364306415285659e-04, -1.872861283057132e-03, -9.364306415285659e-04, 2.201378129763104e+00, 4.402756259526209e+00, 2.201378129763104e+00, -1.289255664090611e+01, -2.578511328181222e+01, -1.289255664090611e+01, 1.933294485262748e+03, 3.866588970525495e+03, 1.933294485262748e+03, 9.636895525355279e-10, 1.927379157244350e-09, 9.636895525355279e-10, 9.841562249707718e-18, -1.740874276634848e-16, 9.841562249707718e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.911072098095597e-03, 2.911072098095577e-03, 1.156606598474184e-02, 1.156606598474184e-02, -1.155290823816465e-01, -1.155290823816465e-01, 1.416096605622251e-01, 1.416096605621939e-01, -1.272481113117374e+00, -1.272481112240240e+00, -2.925370667781380e-07, -2.925370667755115e-07, -7.729234110010820e-19, -7.729379511308610e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
