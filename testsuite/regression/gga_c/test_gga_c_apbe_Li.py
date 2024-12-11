
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_apbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.649638401824437e-02, -3.982333276923552e-02, -2.347354537672937e-03, -1.468743760766970e-02, -1.050071025521184e-03, -5.425313275020383e-09, -1.283695372222837e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_apbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.147207762813762e-01, -1.146000446942027e-01, -9.867728910512186e-02, -9.858842977398079e-02, -1.252065589653689e-02, -1.252505949484807e-02, -2.459048111101621e-02, -9.265650076593812e-02, -5.343844395139685e-03, 2.854690522816329e-01, -3.511466660663699e-08, -3.529105805756179e-08, -8.101226395447469e-16, -9.591462281528795e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_apbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.815752603460336e-05, 9.631505206920672e-05, 4.815752603460336e-05, 1.520969725681395e-04, 3.041939451362789e-04, 1.520969725681395e-04, 2.476262920115696e-03, 4.952525840231388e-03, 2.476262920115696e-03, 3.850842432607036e+00, 7.701684865214072e+00, 3.850842432607036e+00, 9.285840493421754e+00, 1.857168098684350e+01, 9.285840493421754e+00, 1.196848350603562e-04, 2.393696701107906e-04, 1.196848350603562e-04, 1.145271653973376e-06, 2.290694101076713e-06, 1.145271653973376e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
