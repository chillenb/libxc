
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.672793734505156e-02, -6.896498763574202e-02, -1.308032100366321e-01, 1.677724006257281e-02, -5.780269126113900e-02, -3.228149230304574e-04, -1.730232014659572e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.689402191500964e-02, -5.603511425380245e-02, -4.716829119372767e-02, -4.666892514304874e-02, -4.774942634370191e-02, -4.064395254065925e-02, 3.174115192211870e-02, -1.799298997262614e-02, 1.677731137960869e-02, -8.119806781196982e-03, -2.193834791581557e-02, -2.186609208092342e-02, -7.274757403837339e-04, -7.516522783245633e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.434257453235339e-04, 0.000000000000000e+00, 3.460078294353291e-04, 2.549672255902727e-03, 0.000000000000000e+00, 2.536511857302832e-03, -1.229013524576512e+00, 0.000000000000000e+00, -1.237156431765228e+00, 4.972935087413775e+00, 0.000000000000000e+00, 3.448701208205041e+02, 2.866834069475813e+03, 0.000000000000000e+00, 2.125516414108316e+06, 9.779768749123264e+00, 0.000000000000000e+00, 4.704039018296905e+02, 3.325488498181473e+01, 0.000000000000000e+00, 4.448496756108179e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.353882106995725e-03, -2.387201776968643e-03, -4.250604900727823e-03, -4.270583837459617e-03, -1.887471468320439e-02, -2.002661291481736e-02, -3.472505304230682e-01, -5.317318012104792e-03, -7.192124179940829e-01, -8.694666731961822e-04, -1.452629785671704e-04, -6.834748234173634e-03, -4.037676922090609e-09, 8.452239096440000e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
