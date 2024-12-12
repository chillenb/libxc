
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.781171078954909e-02, -5.725102115883426e-02, 8.927006556151834e-03, -1.039513532614195e-04, 5.706557175194360e-08, 1.636472863754987e-02, 3.490958209990071e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.710510511865072e-02, -8.683315945976006e-02, -6.436878533383451e-02, -6.402120552054590e-02, -7.176746033650150e-02, -7.129320499308686e-02, -6.400010201737816e-03, 5.312662070926171e-01, 1.146443827471666e-02, 3.213786077883324e-01, 2.263222112333413e-02, 2.233697043080720e-02, 3.421084117826404e-04, 8.838513469570094e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.412708980890810e-05, 0.000000000000000e+00, 2.421297250085603e-05, -1.349225138972558e-04, 0.000000000000000e+00, -1.350502792239279e-04, 1.952345333558813e-02, 0.000000000000000e+00, 1.739766414885255e-02, 7.690054626895138e+00, 0.000000000000000e+00, 8.821921371962393e+00, -6.689820719662379e+01, 0.000000000000000e+00, -3.318671766924803e+05, -5.639663743422666e-01, 0.000000000000000e+00, -7.409775612191295e+01, -3.684330450293112e+00, 0.000000000000000e+00, -1.612853253864596e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.967962407727972e-04, -2.122384770136134e-04, 3.909879930728599e-03, 3.905541732527527e-03, 7.653235803644712e-03, 8.407962097226558e-03, -2.759086379073410e-01, 1.040926708365400e-03, 1.599854415662642e-01, 1.395565818086916e-04, 5.967042893052173e-07, 1.091703388064808e-03, 5.106358027030929e-15, 7.035872782433667e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
