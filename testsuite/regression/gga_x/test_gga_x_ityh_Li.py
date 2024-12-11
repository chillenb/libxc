
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ityh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.685917512975307e+00, -1.170119054921031e+00, -2.609141721466337e-01, -7.323555238178965e-02, -9.812870377883027e-03, -7.206693265920491e-05, -4.950651026357798e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ityh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.140284167736527e+00, -2.142389442800043e+00, -1.429789126503317e+00, -1.431134822069540e+00, -2.885981646780387e-01, -2.885397870588355e-01, -1.151675796596350e-01, -1.252355610314151e-04, -1.785137542625547e-02, -3.999987436450727e-09, -1.455995262028283e-04, -1.424521839012475e-04, -1.191999292489478e-09, -4.282697259569788e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ityh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.476023272349307e-04, 0.000000000000000e+00, -2.468070178819249e-04, -8.876261392078441e-04, 0.000000000000000e+00, -8.849665953312634e-04, -5.099900630630522e-02, 0.000000000000000e+00, -5.094619657439668e-02, -1.190446608651492e+00, 0.000000000000000e+00, -4.107233686610825e-04, -1.524006072546700e+00, 0.000000000000000e+00, -9.107755816899282e-08, -4.760149287327382e-04, 0.000000000000000e+00, -4.454618766462165e-04, -2.266593011881376e-08, 0.000000000000000e+00, -1.096493776397239e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
