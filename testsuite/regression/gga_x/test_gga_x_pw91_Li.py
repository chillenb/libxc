
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pw91_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.802292587779567e+00, -1.290406189213285e+00, -4.190057744484081e-01, -1.605410020570628e-01, -8.050461088072250e-02, -3.150226553409883e-04, -4.422816966140539e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pw91_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.242991216316388e+00, -2.245109550586391e+00, -1.526448530223716e+00, -1.527806657697138e+00, -3.845530040923101e-01, -3.846999052813779e-01, -2.050857665463204e-01, -1.125679186039283e-03, -7.604586523544220e-02, -8.228053776626846e-08, -1.325445299968622e-03, -1.250557246656208e-03, -2.085895355317184e-08, -8.956448704498537e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pw91_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.725560775077192e-04, 0.000000000000000e+00, -2.716573672272114e-04, -1.018983044059429e-03, 0.000000000000000e+00, -1.015807581698872e-03, -8.443668813048189e-02, 0.000000000000000e+00, -8.426810311538854e-02, -4.398698837605233e+00, 0.000000000000000e+00, 7.285061303611574e+00, -6.847775754359431e+01, 0.000000000000000e+00, 4.175689889080783e+01, 7.439935701029230e+00, 0.000000000000000e+00, 6.932018228464640e+00, 3.029810382347581e+01, 0.000000000000000e+00, 4.331544084885138e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
