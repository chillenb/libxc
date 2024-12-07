
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.094262441521900e+00, -7.858419245250905e-01, -2.482221221293842e-01, -1.070772991776995e-01, -4.779768311609019e-02, -9.793150716786237e-04, -3.752345196689400e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.395176870696341e+00, -1.396261176081685e+00, -9.716586327043545e-01, -9.723382270650249e-01, -2.251830819592592e-01, -2.251811771938342e-01, -1.413407633343317e-01, -1.002196581849682e-01, -4.964437221536998e-02, 4.054855024929208e-01, -3.694389925272628e-03, -3.513791946052989e-03, -1.589217604651262e-07, -7.321582313621039e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.180717029426852e-04, 7.967326140990596e-05, -1.175508930916872e-04, -4.603048133873848e-04, 2.615416240187202e-04, -4.584603647232310e-04, -5.452222431079341e-02, 7.580909613364026e-03, -5.447175609367145e-02, 7.125211998271683e-01, 6.521928193662712e+00, 2.282966016263292e+01, -3.000537107897898e+01, 2.364302250386845e+01, 2.755080359439895e+02, 1.962081966420259e+01, 3.464085846831062e-04, 1.843411104329134e+01, 2.189947001140764e+02, 3.213906681076925e-06, 3.359107612234272e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
