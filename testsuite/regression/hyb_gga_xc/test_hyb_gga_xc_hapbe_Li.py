
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hapbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.502570110086076e+00, -1.078982565857823e+00, -3.424903332002582e-01, -1.402166743561854e-01, -6.684625584442727e-02, -1.643713515263654e-02, -3.070869818578925e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hapbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.901622872574046e+00, -1.903152719891270e+00, -1.309201763739307e+00, -1.310145614225288e+00, -3.439638360869021e-01, -3.441359921154665e-01, -1.833891106569038e-01, -9.502772996533562e-02, -6.655520991409215e-02, 2.277115267323729e-01, -2.197214428576159e-02, -2.181396443403351e-02, -4.433245338972561e-04, -3.151634089672010e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hapbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hapbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.908708757279812e-04, 9.442806658541296e-05, -1.900571510064033e-04, -7.720404934960176e-04, 3.150543638182370e-04, -7.690871792659769e-04, -5.638224449783860e-02, 4.116822188154279e-03, -5.621943851194123e-02, -6.283224829773040e-01, 6.161378741747470e+00, 2.893038566836544e+00, -4.887050080784452e+01, 1.485734478948655e+01, 6.228800562862340e+00, -1.906058956101157e-01, 1.915045909854491e-04, -1.779822954765505e-01, -8.734610088907392e-01, 1.832555280861370e-06, -1.250269779088983e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
