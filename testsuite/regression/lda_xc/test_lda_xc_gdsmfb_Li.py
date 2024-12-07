
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_gdsmfb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_gdsmfb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.824102284628247e+00, -1.285156292748977e+00, -3.388008446267026e-01, -1.749223334171319e-01, -7.323877188498651e-02, -1.820599182080379e-02, -3.791920361842860e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_gdsmfb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_gdsmfb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.409421157094257e+00, -2.411314365311914e+00, -1.693713496703284e+00, -1.694923483280362e+00, -4.420373703194696e-01, -4.419182353057282e-01, -2.301124051072689e-01, -1.532029330732434e-01, -9.611094097629221e-02, -7.840606607154346e-02, -2.382574602344961e-02, -2.380991128864541e-02, -5.090256068716177e-04, -4.891209544956707e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
