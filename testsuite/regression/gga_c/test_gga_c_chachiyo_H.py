
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_chachiyo_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.160589444331602e-02, -1.495842294087652e-02, -5.900691935575716e-03, -2.060366154006817e-05, -9.618269836711267e-95]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_chachiyo_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.666097558686197e-02, -1.714687681461037e+03, -3.960927548661106e-02, -1.227842850218188e+03, -2.321847436601869e-02, -3.883295228023385e+02, -2.210955210452841e-04, -9.800746879094880e-01, -1.694868935062546e-92, -4.353324682437377e-90]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_chachiyo_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.076283168868185e-02, 4.152566337736370e-02, 2.076283168868185e-02, 1.140413320752818e-02, 2.280826641505636e-02, 1.140413320752818e-02, 3.835924920638914e-02, 7.671849841277828e-02, 3.835924920638914e-02, 2.225723691185535e-02, 4.451447382371070e-02, 2.225723691185535e-02, 9.697274006291094e-87, 1.939454801258219e-86, 9.697274006291094e-87]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
