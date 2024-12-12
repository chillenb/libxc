
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2_rev_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.952703418319356e+00, -1.368982480431770e+00, -3.729095610987900e-01, -1.757311502122978e-01, -7.621995576971841e-02, -1.712967268058199e-02, -3.200241284136042e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2_rev_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.550773962812983e+00, -2.553153190119864e+00, -1.756130081540227e+00, -1.757927653805071e+00, -3.436366586828165e-01, -4.232015614460411e-01, -2.314783523350135e-01, -2.177864853405505e-02, -8.838537595691053e-02, -6.916761180832591e-04, -2.289804952002817e-02, -2.273321948180551e-02, -4.620011818408396e-04, -3.284407897608138e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_rev_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.867158709286660e-05, 0.000000000000000e+00, -8.818299524703828e-05, -3.628636009495724e-04, 0.000000000000000e+00, -3.578683886885035e-04, -1.397413497845986e-01, 0.000000000000000e+00, -3.619269134513597e-02, -1.517402801895175e+00, 0.000000000000000e+00, -1.938331164660413e-01, -2.897511075689085e+01, 0.000000000000000e+00, -2.017304669371944e+00, -1.971713497101614e-01, 0.000000000000000e+00, -1.839282898565867e-01, -9.035686113799892e-01, 0.000000000000000e+00, -1.590282313310667e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_rev_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([9.330910633603736e-06, 9.045070206782281e-12, 6.639794209542826e-05, 1.325920696422546e-17, 2.456902745323119e-02, 7.591025002014500e-11, 6.200360205124246e-03, 9.761105659045496e-18, 6.705111626235770e-07, 3.162022506894115e-10, 9.216792542305215e-22, 3.478072782797176e-18, -5.562233065891707e-38, 1.295246273939376e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
