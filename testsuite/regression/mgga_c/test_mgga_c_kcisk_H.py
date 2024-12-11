
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_kcisk_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.507883190625256e-15, -3.118057543767166e-02, -2.518504806175199e-02, -1.327193674949381e-02, -1.569791179574516e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_kcisk_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.219601684207910e-02, -2.242845691370779e-01, -3.508782721215671e-02, -2.146767187768154e-01, -2.872192988328922e-02, -1.713347950604392e-01, -1.571594930843385e-02, -8.486840964027165e-02, -2.001663684282091e-03, -7.156556982179594e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.316919874077979e+00, 2.223666092188941e-02, 8.608116618147828e+13, 5.507333298752828e-02, 3.290366312653045e-02, 8.596311204654975e+13, 2.518847312783641e-01, 2.104229440293075e-01, 8.612673461976180e+13, 2.573688233561270e+01, 4.350635889410380e+01, 7.269160365218112e+13, 5.691073528152057e+12, 1.138214704961406e+13, 7.838285363878170e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.140833624679643e+00, -3.995554316157371e-04, -1.776359381880728e-38, -3.990074693494971e-04, -3.171872743719510e-37, -3.997669429258967e-04, -4.440135579057598e-34, -6.330126426738284e+00, -2.646759132179996e-27, -6.330142115625751e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
