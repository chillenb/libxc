
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.217394393806259e+00, 4.928103969598261e+00, 3.489669659060258e+00, 3.292116582640138e-02, 7.341060633769032e-02, 5.285880799345448e+01, 8.785715181475955e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.217239417951498e+00, -4.217145241007243e+00, 3.373541805637733e+00, 3.401853260750157e+00, 3.782028747687363e-01, 3.765126419991184e-01, 7.269935071524374e-02, -3.386363269636171e-01, -4.006247107134852e-02, -1.210226358618995e+00, -3.357579634641856e-01, -3.475308790934789e-01, -1.576047662316221e-01, -1.185586352313799e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.927192691708965e-02, 0.000000000000000e+00, 1.921992702677494e-02, 3.854342399576002e-02, 0.000000000000000e+00, 3.838496380133952e-02, 4.614896372377465e-01, 0.000000000000000e+00, 4.620789158761048e-01, 1.914057079195393e+01, 0.000000000000000e+00, 8.699790635302614e+03, 3.115351564287113e+02, 0.000000000000000e+00, 2.454376727176486e+09, 7.481638617199318e+03, 0.000000000000000e+00, 7.647285960080892e+03, 9.151270127775303e+08, 0.000000000000000e+00, 2.292358375079883e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk4_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([5.169830032053591e-08, 5.216466590635551e-08, 6.199699490961624e-02, 6.219354725802997e-02, 1.666666666666667e-01, 1.666666665761873e-01, 4.973369457189177e-02, 1.666666666666298e-01, 4.780708338509948e-02, 0.000000000000000e+00, 1.666666666666667e-01, 1.666666666666668e-01, 1.666666666666665e-01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
