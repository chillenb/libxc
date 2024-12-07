
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tau_hcth_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.869103723650425e+00, -1.286350131435918e+00, -4.885089292603628e-01, -1.705260103726849e-01, -7.486183259459506e-02, -4.380169430221372e-02, -8.191972192906768e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tau_hcth_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.580333483835421e+00, -2.582779583019063e+00, -1.721295137238357e+00, -1.722866636558667e+00, -1.883106951162828e-02, -1.905926555282604e-02, -2.345503133988389e-01, -5.558462358343626e-02, -4.353577176888247e-02, -1.770540632894120e-03, -5.842207036443561e-02, -5.801032899807369e-02, -1.182627504256098e-03, -8.407415004210099e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.526864684906716e-04, 0.000000000000000e+00, 1.524702462464635e-04, -3.981932478208493e-07, 0.000000000000000e+00, 5.024892149418509e-07, -2.454720096235979e-01, 0.000000000000000e+00, -2.428174974110566e-01, 3.414560735949559e+00, 0.000000000000000e+00, -1.547195723462817e+00, -1.032176338293542e+02, 0.000000000000000e+00, -9.890886148200943e+00, -1.572391977956140e+00, 0.000000000000000e+00, -1.468286722710493e+00, -7.200193534288819e+00, 0.000000000000000e+00, -1.030633388656866e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.657096916750760e-05, 3.762537561572277e-05, 1.019635882285959e-03, 1.009791934245622e-03, -2.313724886345935e-02, -2.420689161058165e-02, 3.065586336552956e-03, -2.786127548641670e-08, -7.628293451364419e-02, -1.447982697407562e-14, -3.144104816356411e-13, -3.192933108956261e-08, 9.770849023959279e-24, -3.712834149743350e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
