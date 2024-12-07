
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_b97_3c_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_3c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.903342652053652e+00, -1.340288690323653e+00, -4.689561637385050e-01, -1.714885728744173e-01, -8.817959065816856e-02, -1.749888677356955e-02, -3.812453279454304e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_b97_3c_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_3c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.515074189389063e+00, -2.517337744136009e+00, -1.723884829428739e+00, -1.725368265229269e+00, -2.738165019262634e-01, -2.747333146858726e-01, -2.278086744825836e-01, 1.016294021572311e+00, -4.847484959552804e-02, 6.492827762213784e-01, -2.537614037713243e-02, -2.388016722182666e-02, -9.363144322873891e-04, 6.818386652441567e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_b97_3c_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_b97_3c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.414719764753363e-05, 0.000000000000000e+00, -1.380357177681368e-05, -2.537289128728123e-04, 0.000000000000000e+00, -2.519090943974227e-04, -1.729385365697013e-01, 0.000000000000000e+00, -1.725511641373982e-01, -2.840327016301923e-01, 0.000000000000000e+00, 1.433578289214975e+02, -1.489912918590476e+02, 0.000000000000000e+00, 1.716475528134653e+04, -6.630613777948825e-01, 0.000000000000000e+00, -3.957020561747319e-01, -8.223587349495395e+00, 0.000000000000000e+00, 2.944594989138450e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
