
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037422780933269e+00, -1.413111879273508e+00, -3.261586294489217e-01, -1.841702389938188e-01, -7.184056513635365e-02, -6.002007206042661e-03, -1.816354230794169e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.719570985102992e+00, -2.722014612766085e+00, -1.891943287425513e+00, -1.893636405878515e+00, -2.586900247225450e-01, -4.634039553401170e-01, -2.454626670545811e-01, -1.124692927991320e-02, -9.908290499162338e-02, -1.105765751029703e-04, -9.748732003388378e-03, -1.184713627483785e-02, -5.909662106884107e-05, -4.789669541835881e-21]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.248400047533717e-06, 0.000000000000000e+00, 7.224577549782157e-06, 4.449491155959841e-05, 0.000000000000000e+00, 4.561034488281656e-05, -2.562531855078651e-01, 0.000000000000000e+00, 1.227505756795500e-02, -3.775004467749737e-01, 0.000000000000000e+00, 2.805042356183251e+01, 7.211252343964341e+00, 0.000000000000000e+00, 2.662357904948127e+04, 2.086075920776739e+01, 0.000000000000000e+00, 2.519790219596326e+01, 4.120001825222969e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rscan_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.023315850765179e-07, 5.665485943512881e-19, 2.243558141248821e-05, 4.879253099053600e-32, 6.454790457243444e-02, 5.913609833472705e-17, 1.710919367334127e-02, 4.526913255355321e-36, 4.230847841847962e-13, 1.170841690618642e-35, 9.886941293759303e-07, 2.626418651116829e-36, 6.762225542634526e-13, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
