
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pbe4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.391782562737541e+01, 1.006610330076505e+01, 3.596001857128491e-01, 9.955306717374637e-02, 3.301555056170737e-02, -5.964996360553251e-04, -2.129036739426481e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pbe4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([7.333383221830969e+00, 7.384601745515153e+00, -2.759981640724585e-01, -2.803149076654133e-01, 1.915906856313950e+00, 1.910055004099277e+00, 1.457963747000263e-01, -8.998732953567271e-04, 1.072909277832917e-01, -9.149372655885851e-07, -9.938378898519840e-04, -9.799988191788388e-04, -4.082036452002332e-07, -2.063033314903136e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pbe4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.716247924602477e-02, 0.000000000000000e+00, 2.704281489761768e-02, 8.979751571464351e-02, 0.000000000000000e+00, 8.962820829799732e-02, -6.377354642604459e-01, 0.000000000000000e+00, -6.355856450764205e-01, 9.819663422269196e+00, 0.000000000000000e+00, -5.763346590528717e-02, -1.143679902212732e+02, 0.000000000000000e+00, -1.167677198219340e-02, -6.159746769188467e-02, 0.000000000000000e+00, -5.709879579775357e-02, -5.677673203696868e-03, 0.000000000000000e+00, -5.777555068879350e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
