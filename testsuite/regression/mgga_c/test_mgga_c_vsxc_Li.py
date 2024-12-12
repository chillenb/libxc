
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_vsxc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.418183836832567e-02, -4.136380766676889e-02, 1.311797489679860e-02, -1.416508826627017e-03, -7.539581051393691e-09, -1.246270935329246e-04, -2.218351946204616e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_vsxc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.400237653467507e-02, -5.415032962402576e-02, -4.854628934342153e-02, -4.864053056117624e-02, -1.271222342952946e-01, -1.289577167310012e-01, -3.026921204457235e-02, 5.196472235061570e-03, -3.861566245871621e-02, 7.946883320670015e-06, -6.846190090333143e-04, -2.517757343347612e-04, -8.443067783386873e-12, -1.585312843128338e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.052095253958711e-04, 0.000000000000000e+00, 2.055741743654741e-04, 3.463400524936447e-04, 0.000000000000000e+00, 3.489950187522158e-04, 1.471476871957014e-01, 0.000000000000000e+00, 1.536895138034661e-01, 3.634931550548016e+01, 0.000000000000000e+00, -3.867290547204872e+01, 2.253334973390184e+02, 0.000000000000000e+00, -4.868437560758004e+03, 1.506298719700832e+00, 0.000000000000000e+00, 3.171565085902198e+00, 2.021550591943264e-06, 0.000000000000000e+00, 3.062210585163816e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.468523744001653e-02, -1.472521060446068e-02, -9.197696411407342e-03, -9.264125054120019e-03, -2.964331074549361e-02, -3.196164107916871e-02, -1.311055721553007e+00, 3.907098455364414e-05, -5.388796548243873e-01, 2.638311636635121e-07, 1.020805933454304e-06, -3.719485812496460e-05, 1.682960037228807e-17, -1.335833195331877e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
