
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.073673989778461e-01, -4.880803213311879e-01, -1.816235450495002e-01, -5.814590280119643e-02, -3.307630256271601e-02, -3.436163441638698e-02, -5.848435809027615e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.510662580766755e-01, -8.541717942676309e-01, -4.659407938188344e-01, -4.651804325100100e-01, -2.211906941255831e-01, -2.206526399136211e-01, -8.114664663748168e-02, -4.337481078483210e-02, -2.219201905999565e-02, -1.390050765899119e-03, -4.607933282696026e-02, -4.525065935037918e-02, -9.284986788189228e-04, -3.658948685982597e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.719241392364564e-05, 0.000000000000000e+00, -9.686434583945348e-05, -4.843719435027036e-04, 0.000000000000000e+00, -4.824523345542097e-04, -2.255426445893328e-01, 0.000000000000000e+00, -2.255416180731177e-01, -1.538831236580750e+00, 0.000000000000000e+00, -4.635092125112157e-01, -8.939785182823029e+01, 0.000000000000000e+00, -2.976447585577326e+00, -1.982485299985978e-04, 0.000000000000000e+00, -4.397478109238108e-01, -1.358573897572904e-10, 0.000000000000000e+00, -5.867815922570135e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.930691979364243e-03, 6.254846006163292e-03, -2.700746545177632e-02, -2.716706125539010e-02, -3.857138472575722e-03, -3.959651962695479e-03, 1.026312736564376e-01, -4.168412036591906e-05, -1.834494445520515e-01, -8.534771703338501e-09, -2.072098175370550e-08, -4.499159496359889e-05, -1.158475360276895e-19, -5.278207091827588e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
