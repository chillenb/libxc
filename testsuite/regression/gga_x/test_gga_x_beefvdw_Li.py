
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_beefvdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.841398169795189e+00, -1.318249083232812e+00, -4.633862659538325e-01, -1.645702490219179e-01, -8.713616966269781e-02, -2.129879106124885e-02, -3.978661426364077e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_beefvdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.315555993835134e+00, -2.317838641440113e+00, -1.537270454535278e+00, -1.538736791851229e+00, -4.564976529226259e-01, -4.571269475915725e-01, -2.129889776407633e-01, -2.708489764621838e-02, -6.405345132613034e-02, -8.599191115950049e-04, -2.847814079702622e-02, -2.827261243013336e-02, -5.743775305969068e-04, -4.083301132502891e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_beefvdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.375011573490171e-04, 0.000000000000000e+00, -2.365276523097585e-04, -1.157507395570193e-03, 0.000000000000000e+00, -1.153422491522501e-03, -7.825425057428885e-02, 0.000000000000000e+00, -7.783834483528378e-02, -3.163759053423737e+00, 0.000000000000000e+00, -1.843255436619324e-01, -1.140684803744538e+02, 0.000000000000000e+00, -1.179255267609865e+00, -1.873101051579926e-01, 0.000000000000000e+00, -1.749163049605045e-01, -8.584545551233970e-01, 0.000000000000000e+00, -1.228789423798830e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
