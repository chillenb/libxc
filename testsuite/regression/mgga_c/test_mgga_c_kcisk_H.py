
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
    ref_tgt = [-4.954789356580217e-11, -1.186154993669062e-11, -5.324274007696712e-12, -4.005120838907043e-13, -1.159699320455744e-11]
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
    ref_tgt = [-3.219601684951400e-02, -2.242845691366765e-01, -2.218552218882594e-02, -2.090293584827193e-01, -1.482906614918022e-02, -1.486322876592764e-01, -2.220672788680544e-03, -3.280217303483997e-02, -4.736030152857141e-06, -9.920875264317733e-05]
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
    ref_tgt = [1.316919874112194e+00, 2.223666092259725e-02, 8.608116618147828e+13, 2.747987054641393e-02, 1.518690123666043e-02, 8.596311204654975e+13, 8.636185478076266e-02, 6.943779575790908e-02, 8.612673461976172e+13, 6.665568687263903e-01, 1.062591224946539e+00, 8.613032435712297e+13, 1.009184864866658e+01, 2.009733486995876e+01, 8.613054221073778e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
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
    ref_tgt = [-3.140833619887608e+00, -3.995554316157371e-04, -4.724584785679618e-02, -3.990074693494971e-04, -2.971029184283762e-02, -3.997669429258967e-04, -4.367530934995582e-03, -3.997836051175632e-04, -6.913882800105558e-06, -3.997846163109039e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
