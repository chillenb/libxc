
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sfat_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.590267423842571e+00, -1.079254281057896e+00, -1.993356395492061e-01, -5.473883648820900e-02, -7.500326800519080e-03, -5.914124656723177e-05, -4.091434833117526e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sfat_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.056307217652926e+00, -2.058405387459034e+00, -1.349410434134879e+00, -1.350743533171446e+00, -2.585660501456384e-01, -2.585583803770857e-01, -8.783664923070868e-02, -1.026508006866747e-04, -1.335245343514099e-02, -3.305743452137759e-09, -1.192459823862101e-04, -1.166805663088918e-04, -9.851192069848016e-10, -3.539411631185194e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sfat_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.130375075546654e-04, 0.000000000000000e+00, -2.123378037505195e-04, -7.797548111682219e-04, 0.000000000000000e+00, -7.774220158037582e-04, -2.426858464550116e-02, 0.000000000000000e+00, -2.419164545359150e-02, -7.456860154435322e-01, 0.000000000000000e+00, -5.130547780931471e-06, -1.318800844068180e+00, 0.000000000000000e+00, -3.384443755191301e-11, -6.362561300874956e-06, 0.000000000000000e+00, -5.772050429803117e-06, -4.904077351848070e-12, 0.000000000000000e+00, -1.792975644698340e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
