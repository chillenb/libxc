
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_cap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_cap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.786400292022340e+00, -1.271228651614646e+00, -4.310319789185538e-01, -1.597985220902186e-01, -7.921814118166545e-02, -4.392142892030479e-01, -4.658592261870819e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_cap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_cap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.257870622585687e+00, -2.259990980725476e+00, -1.537344701186638e+00, -1.538718115379607e+00, -2.791536571834068e-01, -2.787777065011713e-01, -2.060849870591006e-01, 1.013206282186036e-01, -6.866669883311828e-02, 5.775076367178143e-02, 1.005177782523763e-01, 1.024405353446395e-01, 6.046369751466379e-02, 5.431011107452368e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_cap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_cap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.107361288489257e-04, 0.000000000000000e+00, -2.100229903340617e-04, -8.268682147503996e-04, 0.000000000000000e+00, -8.242015634117302e-04, -1.432851662517276e-01, 0.000000000000000e+00, -1.433791092046532e-01, -3.335615002301833e+00, 0.000000000000000e+00, -6.613901095522372e+03, -8.087142338385756e+01, 0.000000000000000e+00, -4.669021118824539e+08, -5.662294266280681e+03, 0.000000000000000e+00, -5.711344307823478e+03, -1.507402097033215e+09, 0.000000000000000e+00, -4.682019323391347e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
