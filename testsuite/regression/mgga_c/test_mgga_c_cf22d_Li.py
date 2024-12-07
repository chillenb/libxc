
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cf22d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.204401075112164e-01, -7.651678134693722e-02, 1.350199214350398e-02, -3.405271925996166e-02, 4.309223716588094e-03, -1.333776715303643e-02, -3.309387292994891e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cf22d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.550394357449999e-01, -2.547636558406089e-01, -2.201167500928329e-01, -2.199657939550189e-01, 8.226705332103655e-02, 8.226577437745698e-02, -7.356759901823032e-02, -2.284365001126445e-01, 6.443468353770535e-03, 1.634058970235388e+00, -1.676496391333800e-02, -1.695309093380731e-02, -3.893153579584796e-04, -5.712657444360053e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.407630428346925e-04, 2.815260856693850e-04, 1.407630428346925e-04, 1.602869983612701e-04, 3.205739967225402e-04, 1.602869983612701e-04, 1.388017978633277e-02, 2.776035957266554e-02, 1.388017978633277e-02, 1.016639127769521e+01, 2.033278255539043e+01, 1.016639127769521e+01, 5.082070789002435e+01, 1.016414157800487e+02, 5.082070789002435e+01, 4.630850294192795e-04, 9.261700588594595e-04, 4.630850294192795e-04, 4.432298571982456e-06, 8.864611187982752e-06, 4.432298571982456e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.380997703993969e-03, -4.380997703993978e-03, 1.142063919065687e-02, 1.142063919065686e-02, -2.243663405468912e-02, -2.243663405468909e-02, 3.842462424810738e-01, 3.842462424809904e-01, -1.980632988443425e-01, -1.980632987078155e-01, -2.010397735551192e-08, -2.010397735496851e-08, -5.310701623235286e-20, -5.310378509240201e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
