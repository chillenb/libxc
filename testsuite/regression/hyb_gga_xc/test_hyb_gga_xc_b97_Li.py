
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.515711470695032e+00, -1.085552954179276e+00, -3.608935720820508e-01, -1.416871356330077e-01, -6.784147223022569e-02, -1.416256520412761e-02, -2.730238143022288e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.932406307104853e+00, -1.933970785745128e+00, -1.330568778559058e+00, -1.331573925843083e+00, -3.181812472122447e-01, -3.186769252524635e-01, -1.854779976417673e-01, 3.116721487800994e-01, -5.511221369400261e-02, 2.060597454249025e-01, -1.963717891136093e-02, -1.907622935131179e-02, -5.173224555025784e-04, 6.095100797102815e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.197109333656533e-04, 0.000000000000000e+00, -1.192824005590684e-04, -5.394837915688265e-04, 0.000000000000000e+00, -5.375730796804885e-04, -7.962423731727601e-02, 0.000000000000000e+00, -7.936599170926974e-02, -6.909686307452302e-01, 0.000000000000000e+00, 4.833884108762557e+01, -7.690818742309538e+01, 0.000000000000000e+00, 5.782165803326041e+03, -1.856749964311595e-01, 0.000000000000000e+00, -9.806567540364237e-02, -2.571748587132181e+00, 0.000000000000000e+00, 1.020405104594226e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
