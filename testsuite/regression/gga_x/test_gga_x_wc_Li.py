
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_wc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.791103110422692e+00, -1.275709954875462e+00, -3.966051145161323e-01, -1.599165531281650e-01, -7.645375561287858e-02, -2.053557401639712e-02, -3.838585507159015e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_wc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.254333605525768e+00, -2.256441176890871e+00, -1.543946033934467e+00, -1.545294023215452e+00, -3.523643564080611e-01, -3.523753953107210e-01, -2.057142627652462e-01, -2.608559348087644e-02, -7.633541634299107e-02, -8.296405959784656e-04, -2.742228509965085e-02, -2.722673549742257e-02, -5.541548356679115e-04, -3.939539036692309e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_wc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.275450028600977e-04, 0.000000000000000e+00, -2.268030930637520e-04, -8.235545095560434e-04, 0.000000000000000e+00, -8.210601756689225e-04, -8.555838079004377e-02, 0.000000000000000e+00, -8.546204285457777e-02, -3.683450477528793e+00, 0.000000000000000e+00, -4.666017201827108e-01, -5.602519737601367e+01, 0.000000000000000e+00, -3.157669335564513e+00, -4.717583720556607e-01, 0.000000000000000e+00, -4.415710254460313e-01, -2.299089622503136e+00, 0.000000000000000e+00, -3.291068252759528e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
