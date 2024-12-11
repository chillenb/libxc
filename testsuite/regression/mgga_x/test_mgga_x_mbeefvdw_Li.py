
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbeefvdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.971719462204459e+00, -1.304333844157961e+00, -2.502359739183219e-01, -1.796029723127104e-01, -5.588518544735779e-02, -1.466414910779270e-02, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbeefvdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.719972638618736e+00, -2.722361099131294e+00, -1.991313029226359e+00, -1.993140807322996e+00, -3.430074845212230e-01, -3.435185499837085e-01, -2.420058790674465e-01, -1.869510238514866e-02, -8.315365341010152e-02, -1.256080667675454e-17, -1.961585984743133e-02, -1.951886238528263e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeefvdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.462677147882014e-04, 0.000000000000000e+00, -2.450252685750053e-04, -2.217518002137744e-03, 0.000000000000000e+00, -2.211452367558966e-03, -2.035904274104667e-01, 0.000000000000000e+00, -2.040207815205139e-01, -1.656529684660570e+00, 0.000000000000000e+00, 2.692127310799807e-01, -1.080328375888796e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.176758292730601e-04, 0.000000000000000e+00, 2.552431466735469e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeefvdw_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.314406557624476e-02, 1.311507721758613e-02, 3.701483764529534e-02, 3.702123391017968e-02, 1.704428919943161e-03, 1.834692196065601e-03, 7.003639050765369e-02, 6.253848627225541e-12, 7.234285177922507e-02, 0.000000000000000e+00, -1.960828523354510e-11, 1.977389772425655e-12, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
