
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mn12_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.515287803034068e+00, -1.218654114864221e+00, -1.835071843872635e-01, -1.125398778184093e-01, -9.202138751174900e-02, 9.839812892625610e-03, 2.120147607078950e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mn12_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([6.390895541845507e-01, 6.502783414090022e-01, -1.527676851806737e+00, -1.526229093282145e+00, -6.441510378001096e-01, -6.516348654600513e-01, -4.187960114175748e-01, 1.232581618437109e-02, -8.887009536461327e-02, 4.569240305043150e-04, 1.313144190543638e-02, 1.278189885339711e-02, 3.057869477540661e-04, 2.176272281595750e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn12_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.012257754558680e-03, 0.000000000000000e+00, -4.008507049854756e-03, -5.280897528366170e-03, 0.000000000000000e+00, -5.287209287283171e-03, -1.217063280162515e-02, 0.000000000000000e+00, -1.100950804366504e-02, 3.672729746269061e+01, 0.000000000000000e+00, -3.665639145235511e+00, -2.994661188658961e+02, 0.000000000000000e+00, -2.345673542435423e+01, -3.726440705229918e+00, 0.000000000000000e+00, -3.478179461790459e+00, -1.707459551729900e+01, 0.000000000000000e+00, -2.443952440000558e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn12_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.154160234580424e-02, -4.237770977713562e-02, 1.251652866849328e-01, 1.252648126017249e-01, 8.461157413471353e-02, 8.615834339486514e-02, 5.270647434925992e+00, 2.438923148147088e-05, 8.754580893012424e-01, 5.199432122154528e-09, 1.204790880051721e-08, 2.627959050266486e-05, 7.076150906038584e-20, 5.805383163209055e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
