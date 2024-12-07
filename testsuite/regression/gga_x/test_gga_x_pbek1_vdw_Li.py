
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbek1_vdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbek1_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.794870198776104e+00, -1.284857377800852e+00, -4.311843429232139e-01, -1.600404834850890e-01, -8.191429599784515e-02, -2.277117222310459e-02, -4.255638836812274e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbek1_vdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbek1_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.240784716943533e+00, -2.242923044883912e+00, -1.514448923978165e+00, -1.515823447106438e+00, -3.819722383975103e-01, -3.821644594176176e-01, -2.052512215279121e-01, -2.893415717271587e-02, -7.332335774902954e-02, -9.197804159603320e-04, -3.041815174993349e-02, -3.020063319649702e-02, -6.143627233006096e-04, -4.367561531809667e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbek1_vdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbek1_vdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.593800759787470e-04, 0.000000000000000e+00, -2.584792231448024e-04, -1.043247305453463e-03, 0.000000000000000e+00, -1.039863879916176e-03, -9.355797402315928e-02, 0.000000000000000e+00, -9.336599756614788e-02, -3.988727169011256e+00, 0.000000000000000e+00, -4.293650898572664e-01, -7.854819713210568e+01, 0.000000000000000e+00, -2.748152806555450e+00, -4.362960945564885e-01, 0.000000000000000e+00, -4.074367061566786e-01, -2.000555833650570e+00, 0.000000000000000e+00, -2.863590514030971e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
