
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbeb0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.411309855168113e+00, -1.012111858556563e+00, -3.167948397925163e-01, -1.358551569795824e-01, -6.237123763840238e-02, -1.540837921587698e-02, -2.878940234020134e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbeb0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.799151484222289e+00, -1.800618307242665e+00, -1.244001360411024e+00, -1.244924984625727e+00, -3.236584076923898e-01, -3.237993441690925e-01, -1.775978321622359e-01, -1.240523985035562e-01, -6.623960197357344e-02, 4.401022380089462e-01, -2.059301863347264e-02, -2.044504813897451e-02, -4.156166464915932e-04, -2.954656511869646e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbeb0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.497158155138822e-04, 8.329492296967884e-05, -1.490566865476773e-04, -6.173164956446771e-04, 2.810807527262560e-04, -6.148848253824662e-04, -5.155293423536614e-02, 8.901735816827147e-03, -5.141471208639999e-02, -2.887547631874550e-01, 5.347704059415098e+00, 2.465564648105799e+00, -3.586466372669435e+01, 2.981570130194060e+01, 1.357551114064229e+01, -2.113661037926895e-01, 5.964828680872490e-04, -1.973589276917533e-01, -9.698911874284438e-01, 5.712068395746166e-06, -1.388300680419526e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
