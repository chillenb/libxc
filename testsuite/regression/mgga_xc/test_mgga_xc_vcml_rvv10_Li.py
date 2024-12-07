
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_vcml_rvv10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.925519908960271e+00, -1.373865992660458e+00, -4.090583851162406e-01, -1.821269680672470e-01, -8.323066037987976e-02, -9.533028422783170e-03, -1.451703973314537e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_vcml_rvv10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.460688372880442e+00, -2.462711221190678e+00, -1.701529123100622e+00, -1.702382699464096e+00, -1.206186404301442e-01, -4.879967820146328e-01, -2.380207766692481e-01, -1.169405808957603e-01, -9.028228474255347e-02, 1.036711589145731e+01, -1.038400147791859e-02, -1.533433029244568e-02, -2.066497148682961e-04, 4.513322658446576e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.959621391784040e-04, 9.135805383659526e-05, -1.954056135181695e-04, -7.563597039348933e-04, 2.967835021607628e-04, -7.600717645194270e-04, -5.354621010981404e-01, 7.121840848733407e-03, -3.220498727675920e-02, -7.406717694925540e-01, 5.801383436491545e+00, 4.281053273881981e+00, -4.380414629784455e+01, 3.056840194094232e+01, -2.011204811204490e+10, 7.017192506779226e-01, 8.834464171198456e-04, 1.309247614105058e+00, 3.151578709131949e+00, 1.011868011739185e-05, -8.726644700115591e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.293512431741306e-05, -1.253514383840194e-11, -1.138880500386458e-04, -2.273069135258136e-17, 1.228605050205888e-01, 2.275899580849679e-10, -8.134660675164397e-03, 1.529669417213426e-12, -1.898846746979189e-06, 8.194360668902494e+00, 8.029565526357888e-16, 4.925383620046366e-13, 1.426902962231702e-33, 3.806841371325491e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
