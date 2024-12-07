
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_24_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.176381744449311e+00, -1.535698660126561e+00, -2.534709694528160e-01, -1.937068567575536e-01, -6.280049813071187e-02, -8.490322500744130e-03, -1.577230498468579e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_24_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.620721843063746e+00, -2.623191741896931e+00, -1.800732243754407e+00, -1.801922963998721e+00, -3.789681283737226e-01, -3.833676284375434e-01, -2.418596444545239e-01, -1.147787818448535e-02, -9.675815729182369e-02, -3.639053414828133e-04, -1.127814375876153e-02, -1.198222930136482e-02, -2.270672075999631e-04, -1.727989284982913e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_24_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.182200211330423e-04, 0.000000000000000e+00, -8.154864352953375e-04, -3.084827060880774e-03, 0.000000000000000e+00, -3.076695969670953e-03, -2.388978070176618e-02, 0.000000000000000e+00, -2.412404062032217e-02, -1.269044913346509e+01, 0.000000000000000e+00, 1.555502924139032e+01, -5.998247879477224e+01, 0.000000000000000e+00, 3.919473157685424e+04, -4.102426440935247e-01, 0.000000000000000e+00, 1.390286141557889e+01, -8.695508932926489e-01, 0.000000000000000e+00, 1.774427186045722e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_24_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_24_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.810347332081324e-02, 2.806217970203129e-02, 4.966344905998345e-02, 4.961118663137123e-02, 1.655635797176623e-02, 1.757686637926744e-02, 2.726117872297046e-01, -1.980269954487709e-04, 3.385653858399809e-01, -1.596919163091671e-05, 1.245467294990757e-07, -2.013134038640887e-04, 8.359996994752165e-16, -7.740602380272520e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
