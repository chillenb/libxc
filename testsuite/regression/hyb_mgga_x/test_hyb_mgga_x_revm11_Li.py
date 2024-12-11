
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_revm11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.088473543879806e+00, -6.932761758199022e-01, -8.438955981321120e-02, -2.503664094060359e-02, -2.154986501788712e-03, -7.755020207917398e-05, -3.879813342080731e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_revm11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.677431940811549e+00, -1.679437843657169e+00, -8.881511698255489e-01, -8.900243856173857e-01, -1.528091710608587e-01, -1.540827143594872e-01, -4.847054607075580e-02, -1.338672605072683e-04, -4.986678775713306e-03, -4.319475156171527e-09, -1.569683813618152e-04, -1.521731973684905e-04, -1.287237460858949e-09, -1.794837693647207e-25]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.346217593853549e-04, 0.000000000000000e+00, 2.334024981713525e-04, 7.519436556512515e-04, 0.000000000000000e+00, 7.525872753986811e-04, -5.072483987557367e-01, 0.000000000000000e+00, -5.030578935413116e-01, 1.658883336074379e-01, 0.000000000000000e+00, -4.934411638190359e-03, -1.739700586363892e+01, 0.000000000000000e+00, -3.185075862544248e-05, -2.327323451309285e-06, 0.000000000000000e+00, -5.101638197174405e-03, -6.486054626614616e-16, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.532132892108297e-02, 2.537288544514759e-02, -1.187839446896926e-02, -1.172757196891652e-02, 2.878454163075129e-03, 3.135410341798888e-03, 9.197182238773009e-02, -2.555801867216061e-08, 6.389735408466641e-03, -5.273310801238555e-15, -1.404445401121967e-11, -3.005726953252904e-08, -3.201782509762129e-26, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
