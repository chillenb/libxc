
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lcy_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.528439059448808e+00, -1.007716772332751e+00, -1.336362141313898e-01, -4.603689628327745e-02, -4.363614194575233e-03, -2.046653948655451e-05, -1.408185527353601e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lcy_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.055099246497521e+00, -2.057041448464910e+00, -1.347029076056196e+00, -1.348238224453556e+00, -2.064498169695989e-01, -2.064023588491743e-01, -7.774972597251079e-02, -9.767608255958841e-02, -1.231737663405114e-02, 3.428185821027419e-01, -4.137202752956780e-05, -4.047930350711749e-05, -3.390582240715987e-10, -1.218201988824809e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lcy_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.430982687600820e-04, 9.190971700708733e-05, -1.424967785335221e-04, -5.089906966108849e-04, 2.980993506782570e-04, -5.071124853242002e-04, -9.109314428125962e-03, 6.249948659585063e-03, -9.065772779869046e-03, 3.097731069588313e+00, 6.762268918356340e+00, 3.381133843333535e+00, 1.105699929351096e+01, 2.258698854598489e+01, 1.129349427298843e+01, 1.670939441014411e-04, 3.357174600576258e-04, 1.671650694362001e-04, 1.606543006016998e-06, 3.212885779437900e-06, 1.606543374555789e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
