
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.869416228690269e-02, -7.054029061880135e-02, -8.870980053879741e-02, -5.366990802761760e-03, -2.919158013527212e-03, 1.684223344025229e-02, 2.746108955445133e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.159024917757427e-01, -1.155288652370519e-01, -1.039613590101113e-01, -1.035616600624997e-01, 6.006391778241360e-02, 5.312205014286780e-02, -3.075700106032567e-03, 6.116068634270353e-01, -3.278878851203925e-02, 3.681605805011077e-01, 3.150826552043021e-02, 3.210044848955862e-02, 5.622292285150144e-04, 7.780335612575794e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.332847807476790e-05, 0.000000000000000e+00, -1.399431888001854e-05, -2.538404107789936e-04, 0.000000000000000e+00, -2.517708856956040e-04, -1.403910710094509e-01, 0.000000000000000e+00, -1.253473512719038e-01, 3.140186390170619e+00, 0.000000000000000e+00, -1.158604176202018e+02, -3.072341219987565e+02, 0.000000000000000e+00, -1.007218407482039e+06, -4.659801798670091e+00, 0.000000000000000e+00, -2.244272514689405e+02, -1.584808770583497e+01, 0.000000000000000e+00, -2.578021013336891e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.639436295215546e-03, 2.624156392132367e-03, 3.799575442151907e-03, 3.772528538333233e-03, -2.894214417560021e-02, -2.906882286075235e-02, -9.588712566932032e-02, 2.973413982749231e-03, 2.453115843759229e-01, 4.160008111651655e-04, 6.923923133281505e-05, 3.260848962821263e-03, 1.924212278199548e-09, -1.603775254033525e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
