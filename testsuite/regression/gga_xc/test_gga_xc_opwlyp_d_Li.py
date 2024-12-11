
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_opwlyp_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opwlyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.857226476647054e+00, -1.338957304682480e+00, -4.109100500306118e-01, -1.604445681086480e-01, -8.082160139256889e-02, -1.352228491771729e-01, -5.364387505964138e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_opwlyp_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opwlyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.310656549135374e+00, -2.312611857773399e+00, -1.593942340253405e+00, -1.595137571982795e+00, -4.534558250786777e-01, -4.538222862488663e-01, -2.050879961277758e-01, -1.245404894017554e-01, -7.321225794154854e-02, -3.855291275822424e-02, -3.921056674460223e-02, -3.948059269161205e-02, -7.481255256488932e-03, -6.547553759504101e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_opwlyp_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opwlyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.576649478095687e-04, 5.627936658925066e-06, -2.569270544228419e-04, -9.557724401951899e-04, 3.932004217313052e-05, -9.533643516168141e-04, -7.447535558378054e-02, 5.331479223861343e-02, -7.425705458540015e-02, -4.314106580657771e+00, 5.972491470810053e+00, -1.335357899212608e+03, -7.560319362306194e+01, 3.917367338362383e+01, -4.850322225879306e+07, -1.164860959238457e+03, 4.587948970189730e-01, -1.166712296969010e+03, -1.440005659745348e+08, 0.000000000000000e+00, -4.289618444088374e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
