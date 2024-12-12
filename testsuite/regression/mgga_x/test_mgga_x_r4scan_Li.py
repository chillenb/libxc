
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r4scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.037410250462186e+00, -1.413069449069496e+00, -3.260244393563159e-01, -1.840367849321235e-01, -7.184056172969112e-02, -5.937081563683965e-03, -1.442334923071038e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r4scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.669849803065874e+00, -2.672334538115724e+00, -1.827711850056424e+00, -1.829382592154799e+00, -2.559527083288820e-01, -3.063334379178975e-01, -2.430363744277021e-01, 3.100805914243296e-01, -8.163572429527081e-02, 3.397593803473437e-03, -9.520561243177409e-03, 3.236239318653542e-01, -4.692758558044223e-05, -4.832379935208682e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r4scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.199738153658101e-04, 0.000000000000000e+00, -2.191860742504986e-04, -8.580655967056448e-04, 0.000000000000000e+00, -8.543053436078884e-04, -2.592110331075086e-01, 0.000000000000000e+00, -1.905499243838104e-01, -3.546015311365694e+00, 0.000000000000000e+00, -8.201847943569375e+03, -9.459806665932341e+01, 0.000000000000000e+00, -7.088048681693013e+06, 2.042826382252246e+01, 0.000000000000000e+00, -7.332775467524260e+03, 3.271620982293099e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r4scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r4scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.179023285427562e-02, 1.177999211349683e-02, 1.569416248456439e-02, 1.566797433498272e-02, 6.521663766943833e-02, 4.877112637939469e-02, 1.383417678732403e-01, 1.051097882968143e-01, 2.434745506386621e-01, 2.898769345982571e-03, 1.087234738715994e-10, 1.069075487569362e-01, 1.230165766265162e-19, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
