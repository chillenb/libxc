
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.942963116352000e-02, -1.437943239984782e-02, -2.283061604894332e-02, -1.095168947906772e-01, -3.110586813732363e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [9.195276085076234e-02, -1.116971020860934e+03, 3.646722975361427e-02, -6.291896932048563e-02, -4.398395428720422e-02, -1.948234886594995e-01, 1.439463605735208e-02, -6.028470696278008e-01, -3.958256645503359e-02, -1.400385234620636e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.869326636469199e-01, -1.973865327293840e+00, -9.869326636469199e-01, -7.362488403239763e-02, -1.472497680647953e-01, -7.362488403239763e-02, 1.127925832185372e-01, 2.255851664370744e-01, 1.127925832185372e-01, 3.063925644699714e+02, 6.127851289399429e+02, 3.063925644699714e+02, 1.007800079040857e+08, 2.015600158081713e+08, 1.007800079040857e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.502141591955219e+00, -4.499872847298339e+00, -6.727160967868147e-02, -6.682083321252837e-02, 2.157437102227848e-02, 2.153101639278596e-02, -1.700213335563924e-01, -1.700203151330312e-01, -7.094149879246215e-05, -7.094149919949627e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
