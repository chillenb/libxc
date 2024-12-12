
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mtask_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.238646444889390e+00, -1.552403999107066e+00, -3.465226085991305e-01, -2.014678053591072e-01, -7.893891563133133e-02, -7.285157922048882e-03, -2.995870621576438e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mtask_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.635446114663948e+00, -2.638114859581492e+00, -1.574549354890066e+00, -2.080741877010482e+00, 1.892218417302833e-01, -5.091917396547345e-01, -2.528325049855123e-01, -1.235821017466763e-02, 7.467430653164080e-02, -1.197454800537125e-04, -1.307105841546033e-02, -1.301772211779925e-02, -6.510430631403674e-05, -4.115105412313561e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mtask_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.604675693503460e-03, 0.000000000000000e+00, -1.600154492431453e-03, -7.040206290770857e-03, 0.000000000000000e+00, 5.011698887275618e-05, -8.696798124881339e-01, 0.000000000000000e+00, 1.348792522586202e-02, -2.267490578706260e+01, 0.000000000000000e+00, 3.082201543917023e+01, -1.063127811939755e+03, 0.000000000000000e+00, 2.569116507215534e+04, 2.745400467982043e+01, 0.000000000000000e+00, 2.768764359671449e+01, 4.538836240284605e+04, 0.000000000000000e+00, 9.535825350083949e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mtask_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mtask", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.367996007632605e-02, 8.366797973724134e-02, 1.231421110852389e-01, 1.235549575853903e-11, 2.109061803053689e-01, 5.078096869096636e-11, 8.740466558111649e-01, 1.669945474956759e-12, 2.561394286255890e+00, 1.451701710206498e-06, 1.955038684411582e-12, 1.776176721797819e-12, 3.353026064690781e-30, 3.167510540627929e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
