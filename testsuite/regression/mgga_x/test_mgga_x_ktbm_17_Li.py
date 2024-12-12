
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.162010813019796e+00, -1.514430586737780e+00, -2.931652334196152e-01, -1.934983372526917e-01, -6.820484009328602e-02, -1.069808972916974e-02, -2.002096333411101e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.647227523553874e+00, -2.649783981641904e+00, -1.781181580867745e+00, -1.782518103324715e+00, -3.794992099647955e-01, -3.812594965739809e-01, -2.450122211510347e-01, -1.305863641713966e-02, -8.894077091438185e-02, -4.141153432301533e-04, -1.435828135418042e-02, -1.363226021942187e-02, -2.893559867532420e-04, -1.966410299726468e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.114031751663846e-04, 0.000000000000000e+00, -8.086670025594235e-04, -3.161032342287905e-03, 0.000000000000000e+00, -3.152795711377598e-03, -6.043718596748925e-02, 0.000000000000000e+00, -6.401204785164004e-02, -1.242172004118650e+01, 0.000000000000000e+00, -1.354733953516624e+01, -9.407608209549606e+01, 0.000000000000000e+00, -3.387921899913910e+04, 3.154923434982698e-01, 0.000000000000000e+00, -1.211406602284544e+01, 6.716409440568604e-01, 0.000000000000000e+00, -1.533760138160681e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.403599372591773e-02, 3.399928891361870e-02, 5.306657129858208e-02, 5.303988300037878e-02, 2.078188036154566e-02, 2.261262091403499e-02, 3.545267711940872e-01, 1.734171221713643e-04, 3.432312297297446e-01, 1.380367355327106e-05, -9.599989649470239e-08, 1.764371613551340e-04, -6.457273186436498e-16, 6.690759444982049e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
