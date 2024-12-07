
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.888399402300300e+00, -1.333955207251430e+00, -4.604088136091157e-01, -1.744483946772035e-01, -7.085877025672666e-02, -2.477973468357126e-01, -7.634035856565481e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.493774377721165e+00, -2.503058289141386e+00, -1.671728745361931e+00, -1.674854522537016e+00, -3.034635473819378e-01, -3.032394804965269e-01, -2.243039174944690e-01, -3.062864475922281e-02, -7.917841824471539e-02, -5.219338417916278e-02, 1.050918004838799e-01, 1.081742209712606e-01, 2.586677296121855e-02, 7.136580784317730e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.799543320309157e-05, -8.922271076991997e-05, 2.722523890006327e-05, -2.570009796534085e-04, -4.473368490162634e-04, -2.588210350887310e-04, -5.914234073743390e-02, -1.848267035372310e-01, -5.913728467734522e-02, -4.893437955256541e+00, -4.982566035702528e+00, -1.115957469223784e+03, -5.245899761339927e+01, -1.054878557569859e+02, -7.948768613870429e+07, -5.368587318423434e+02, -6.862643583658710e+03, -5.425442936620044e+02, 7.730329219032523e+07, -1.580479588590362e+09, -1.817772809630770e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
