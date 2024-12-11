
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_opbe_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.542078746274066e-01, -5.981323904379056e-01, -3.706157577965896e-01, -1.474293418800808e-01, -9.021530368843679e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_opbe_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.651198103568306e-01, 1.024779219195553e+00, -7.600436632087966e-01, 5.695636723851028e+01, -4.289629887028868e-01, 3.985280903978462e+01, -1.297718889396822e-01, 6.266727944472631e-01, -1.201271401762096e-02, 2.209667185300624e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_opbe_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.766888250580820e-03, 2.302752131999052e-02, 1.151376065999526e-02, -1.481466914813555e-02, 1.786705441462763e-02, 8.933527207313816e-03, -1.355942487838481e-01, 8.851708938262184e-02, 4.425854469131092e-02, -7.507926619727463e+00, 3.294797517349290e-01, 1.647398758674643e-01, -1.277982141280402e+01, 4.111787984798844e-03, 2.055893991663872e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
