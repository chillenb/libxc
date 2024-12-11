
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b1pw91_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.414692592327843e+00, -1.015210873307397e+00, -3.281431054444442e-01, -1.356621920074272e-01, -6.239869945329237e-02, -9.990426180119487e-02, -4.021049420858051e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b1pw91_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.794477964017344e+00, -1.795940192951189e+00, -1.242853931693916e+00, -1.243776581724600e+00, -2.746076067053840e-01, -2.745205189275380e-01, -1.778642190631383e-01, -1.238912998418923e-01, -6.197443527173673e-02, 3.996928596194447e-01, -2.749907077862962e-02, -2.763112219931352e-02, -5.595927634729266e-03, -4.839823607499420e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b1pw91_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.656397685719473e-04, 7.967326140990596e-05, -1.649601386393936e-04, -6.421137405365326e-04, 2.615416240187202e-04, -6.396936867949603e-04, -8.222162625407556e-02, 7.580909613364026e-03, -8.219631611910708e-02, -4.608838924853931e-02, 6.521928193662712e+00, -1.001431384930851e+03, -4.562487391455512e+01, 2.364302250386845e+01, -3.637726294552729e+07, -8.736500085691138e+02, 3.464085846831062e-04, -8.750441674682310e+02, -1.080001677427962e+08, 3.213906681076925e-06, -3.217208417871041e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
