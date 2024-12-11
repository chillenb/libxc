
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_gaploc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gaploc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.706281392182148e-02, -4.951634857300546e-02, -5.840181323601441e-03, -1.802843691807531e-02, -1.556514392089005e-03, -4.729325365268299e-05, -2.956075473420852e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_gaploc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gaploc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.681897484085975e-01, -1.679630603640843e-01, -2.280349696261318e-01, -2.278889573368935e-01, -1.004818720455368e-02, -1.005693872966392e-02, -2.157299888672715e-02, -1.132961539906899e-01, -2.470409173975431e-03, -2.095328980227590e-02, -1.713022071442548e-04, -1.725926967291748e-04, -9.717200460273618e-08, -1.395611133665707e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_gaploc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gaploc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.597240624441964e-05, 1.319448124888393e-04, 6.597240624441964e-05, 4.807926186397623e-04, 9.615852372795245e-04, 4.807926186397623e-04, 9.576439213445169e-04, 1.915287842689034e-03, 9.576439213445169e-04, 3.057642249351875e-01, 6.115284498703750e-01, 3.057642249351875e-01, 1.708741163245832e+00, 3.417482326491663e+00, 1.708741163245832e+00, 5.154653378277207e-01, 1.030930675655441e+00, 5.154653378277207e-01, 1.321748970806851e+02, 2.643497941613703e+02, 1.321748970806851e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
