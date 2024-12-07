
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_sogga11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.607782828545744e-02, -6.316644219114301e-02, -1.137986069670652e-01, -1.874940934183398e-02, -2.054583008975760e-02, -7.203819741957433e-02, -1.807551451010421e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_sogga11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.436447130124844e-01, -1.434971925931245e-01, 6.240555147525373e-02, 6.245111987004529e-02, 2.284930467890513e-02, 2.278610076613867e-02, -2.090717292989860e-02, -1.291180987759942e-01, -4.526753098365581e-03, -1.019453465879989e+00, -8.844646478803289e-02, -8.945501229238631e-02, -2.126343947388368e-03, -3.120130791298806e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_sogga11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_sogga11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.672742470748695e-05, 1.334548494149739e-04, 6.672742470748695e-05, -3.803289054232464e-04, -7.606578108464928e-04, -3.803289054232464e-04, -3.989016429417581e-02, -7.978032858835162e-02, -3.989016429417581e-02, -4.533367444071237e-01, -9.066734888142474e-01, -4.533367444071237e-01, -4.624047479499818e+01, -9.248094958999636e+01, -4.624047479499818e+01, -9.003737986470775e+00, -1.800747597294155e+01, -9.003737986470775e+00, -9.435915594727334e+01, -1.887183118945467e+02, -9.435915594727334e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
