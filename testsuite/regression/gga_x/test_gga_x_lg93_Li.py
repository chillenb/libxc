
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lg93_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lg93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.781484886198315e+00, -1.282574809452178e+00, -4.197106094618034e-01, -1.591097018578452e-01, -8.224432487263038e-02, -4.119180478082922e-02, -2.163640711066722e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lg93_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lg93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.229524626815494e+00, -2.231769726554365e+00, -1.439865232693390e+00, -1.441295039945973e+00, -4.161825288411279e-01, -4.161966554577296e-01, -2.056105611411315e-01, -3.720432544894249e-02, -8.197814309153684e-02, -2.946296052445107e-03, -3.849428990640659e-02, -3.849360238033492e-02, -2.344521417220034e-03, -1.848610873526514e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lg93_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lg93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.481011931905474e-04, 0.000000000000000e+00, -2.470309272694736e-04, -1.420261901688199e-03, 0.000000000000000e+00, -1.415194423041186e-03, -6.956453148404373e-02, 0.000000000000000e+00, -6.945697304991726e-02, -3.205907015150435e+00, 0.000000000000000e+00, -1.525285888852012e+02, -6.057241745187240e+01, 0.000000000000000e+00, -8.574463115784596e+05, -1.368037381495375e+02, 0.000000000000000e+00, -1.351295846397342e+02, -1.617108162543638e+06, 0.000000000000000e+00, -3.430126942637947e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
