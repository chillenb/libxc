
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lgap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.643462731978893e+01, 8.177078660532539e+00, 7.050052773247970e-01, 1.324396181651304e-01, 2.795055529013773e-02, 1.230318873477218e-03, 4.371568856668977e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lgap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.595368981740986e+01, 2.600134185754485e+01, 1.228864061884565e+01, 1.231006100336669e+01, 8.130610874967397e-01, 8.137551653026884e-01, 2.139231280080237e-01, 1.867736652659034e-03, 3.101759071357320e-02, 1.878692487687977e-06, 2.065334300763277e-03, 2.035400868194921e-03, 8.381734994530765e-07, 4.236054178529440e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lgap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.422106556693236e-03, 0.000000000000000e+00, 2.415752607859172e-03, 7.000029616423093e-03, 0.000000000000000e+00, 6.982670356300707e-03, 1.755999965743532e-01, 0.000000000000000e+00, 1.749000407665838e-01, 3.348844362567604e+00, 0.000000000000000e+00, 0.000000000000000e+00, 3.406351144418569e+01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
