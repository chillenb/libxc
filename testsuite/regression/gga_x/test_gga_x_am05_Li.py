
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_am05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.737906305580707e+00, -1.210955247505001e+00, -3.475992080965358e-01, -1.569707820092236e-01, -6.803786629750382e-02, -8.893648493949742e-02, -2.978164740574511e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_am05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.305369018439147e+00, -2.307478699805981e+00, -1.583682577639545e+00, -1.585077138941285e+00, -3.190247063440919e-01, -3.188233849642520e-01, -2.090371085801536e-01, -3.023701526779606e-02, -7.239859513819157e-02, -6.257217513549873e-03, -3.091061309866296e-02, -3.107700753333741e-02, -5.907621873646541e-03, -5.046004364189955e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_am05_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.850592990566747e-05, 0.000000000000000e+00, -1.840071295367367e-05, -1.593434460421667e-04, 0.000000000000000e+00, -1.586504560334606e-04, -7.004366846764108e-02, 0.000000000000000e+00, -7.006046230172736e-02, -1.401605312896195e-01, 0.000000000000000e+00, -8.311011033238887e+02, -4.008529274029615e+01, 0.000000000000000e+00, -2.617860038419059e+07, -7.248376259435327e+02, 0.000000000000000e+00, -7.250308515255479e+02, -7.637821618640974e+07, 0.000000000000000e+00, -2.255762023001763e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
