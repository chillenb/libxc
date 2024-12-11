
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3p86_nwchem_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.502849901988850e-01, -5.033866469816163e-01, -3.153154960706221e-01, -1.237396727255799e-01, -4.869663037732603e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3p86_nwchem_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.205634680046488e-01, -2.404987151571411e-01, -6.407858807422696e-01, -2.297587360138205e-01, -3.822248464584557e-01, -1.805252493362234e-01, -1.008370206835118e-01, -6.663858525864355e-02, -1.813578218780569e-02, -8.767331898494404e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3p86_nwchem_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.828234277349150e-03, 2.444384645614427e-02, 1.222192322807214e-02, -8.612783827599771e-03, 1.938914019938463e-02, 9.694570099692315e-03, -6.330089271909284e-02, 1.163787976788890e-01, 5.818939883944448e-02, -6.593739679047538e+00, 1.719411018171161e+00, 8.597055090855804e-01, -3.691424158880226e+04, -1.726139350473333e+00, -8.630696752366667e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
