
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.642635658234033e+01, 8.170542629558899e+00, 6.454699201105014e-01, 1.323341098426040e-01, 2.671144645260153e-02, 1.269771739395747e-03, 4.514613911752276e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.592089208936225e+01, 2.596838936061359e+01, 1.232374642242529e+01, 1.234505892688146e+01, 8.250570730400001e-01, 8.250337137314178e-01, 2.135695663748128e-01, 1.925933495000628e-03, 3.374089226962926e-02, 1.940160003679632e-06, 2.129318032189506e-03, 2.098626147115403e-03, 8.655991291391516e-07, 4.374663404283825e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.454727506197945e-03, 0.000000000000000e+00, 2.448471144467044e-03, 6.757651142488213e-03, 0.000000000000000e+00, 6.741683659763834e-03, 1.216943629460067e-01, 0.000000000000000e+00, 1.213459605181679e-01, 3.435278794436041e+00, 0.000000000000000e+00, 1.724823323464136e-02, 2.358519510966643e+01, 0.000000000000000e+00, 3.499317774011256e-03, 1.843167440739446e-02, 0.000000000000000e+00, 1.708678468573852e-02, 1.701499820947332e-03, 0.000000000000000e+00, 1.731433604217697e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
