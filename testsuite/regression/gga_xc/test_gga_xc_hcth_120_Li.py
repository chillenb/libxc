
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_120_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_120", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.916665131618296e+00, -1.345997856509371e+00, -4.609566462114520e-01, -1.755749242660574e-01, -9.039283276486490e-02, -1.453587818345917e-02, -3.346118935498258e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_120_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_120", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.560035117995098e+00, -2.562411706598680e+00, -1.739234708038788e+00, -1.740785696637972e+00, -3.765441331735113e-01, -3.775708980523531e-01, -2.380516357010554e-01, 6.014550776227520e-01, -6.053358623294295e-02, 3.890369745519103e-01, -2.053096333635416e-02, -1.959812257872939e-02, -7.126628899885408e-04, 2.997991851157552e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_120_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_120", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.302717112580138e-05, 0.000000000000000e+00, 3.334300548479758e-05, -2.098881279292094e-04, 0.000000000000000e+00, -2.079498686698935e-04, -1.149633907448167e-01, 0.000000000000000e+00, -1.144594584918346e-01, 2.325419595354993e+00, 0.000000000000000e+00, 1.062962947783263e+02, -1.279877755904277e+02, 0.000000000000000e+00, 1.263770912268136e+04, 5.909758634300801e-01, 0.000000000000000e+00, 7.162509437396514e-01, -1.106998358780263e+00, 0.000000000000000e+00, 2.874445504098921e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
