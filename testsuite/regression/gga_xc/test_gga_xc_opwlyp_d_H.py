
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_opwlyp_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opwlyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.218927501481654e-01, -5.793434100154549e-01, -3.603015555896765e-01, -1.461613518488168e-01, -5.947303918344496e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_opwlyp_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opwlyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.283765204955421e-01, -2.536263398274878e-01, -7.195184653293717e-01, -2.714960714397611e-01, -4.050841112549066e-01, -2.191208227877763e-01, -1.032757169730805e-01, -4.144071456083203e-02, -1.512860039057851e-02, -2.495678472386845e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_opwlyp_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opwlyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.248283173031272e-02, 3.361102578021298e-02, 2.519861456200286e-02, -2.458982458177318e-02, 5.366262806640298e-02, 4.019473757996516e-02, -1.644894284794381e-01, 4.759524524476690e-01, 3.569537930762601e-01, -1.031117672151485e+01, 2.519923419422081e+01, 1.889939366944716e+01, -5.127587980863692e+04, 3.716177783979757e-15, 2.787129669700612e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
