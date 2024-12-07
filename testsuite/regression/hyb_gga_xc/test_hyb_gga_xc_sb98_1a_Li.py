
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_1a_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.505378550866849e+00, -1.066555369004608e+00, -3.244093534241815e-01, -1.212182156058945e-01, -5.928579314207293e-02, -1.208137884853928e-02, -2.189984220819855e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_1a_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.934489523715365e+00, -1.936085193042569e+00, -1.332044422620991e+00, -1.333036617117951e+00, -3.083707389260305e-01, -3.087244285497396e-01, -1.546210124599282e-01, 2.525368450769920e-01, -5.099783395549266e-02, 1.675946635798722e-01, -1.681319358063506e-02, -1.634869700289002e-02, -4.171046186708284e-04, 5.277226913068341e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_1a_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.148892970529980e-04, 0.000000000000000e+00, -1.144971561151120e-04, -4.444196322430077e-04, 0.000000000000000e+00, -4.429999071180674e-04, -6.131003176789471e-02, 0.000000000000000e+00, -6.111818680594840e-02, -4.494312125208909e+00, 0.000000000000000e+00, 4.045681467683457e+01, -6.244097176735342e+01, 0.000000000000000e+00, 4.842117233390411e+03, -1.832277342367513e-01, 0.000000000000000e+00, -1.080412656350103e-01, -2.306187385647764e+00, 0.000000000000000e+00, 8.325835780674028e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
