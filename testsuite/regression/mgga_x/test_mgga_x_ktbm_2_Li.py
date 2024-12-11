
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.884484701132983e+00, -1.262071589857396e+00, -2.867090672993631e-01, -1.729043357857303e-01, -5.965582103542147e-02, -1.287660205458152e-02, -2.374378519481467e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.631132651033855e+00, -2.633686540943354e+00, -1.818689722210555e+00, -1.820331544988113e+00, -3.591967706313013e-01, -3.584998860507762e-01, -2.376491021338107e-01, -1.579478958074326e-02, -7.677191977399933e-02, -5.009326622180990e-04, -1.660912567666929e-02, -1.648850502506565e-02, -3.345942333412974e-04, -2.316307759909905e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.214151719835114e-04, 0.000000000000000e+00, -5.196214623925693e-04, -2.044112872481472e-03, 0.000000000000000e+00, -2.038183858644625e-03, -6.991386502855429e-02, 0.000000000000000e+00, -7.243280173501310e-02, -8.035535889773936e+00, 0.000000000000000e+00, -1.518078208158708e+01, -8.765712594475676e+01, 0.000000000000000e+00, -3.798990225764653e+04, -2.820645299944928e-01, 0.000000000000000e+00, -1.357412729964539e+01, -5.752744040664302e-01, 0.000000000000000e+00, -2.671350642459089e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.707091955897407e-02, 1.705903401144076e-02, 1.999616133474890e-02, 2.000284268728812e-02, -4.249422934156827e-03, -4.372370166320648e-03, 2.002479854452082e-01, 1.941292742206314e-04, -2.318632822231422e-02, 1.547848143653022e-05, 4.189110268462390e-06, 1.974894614315605e-04, 6.984754830503699e-11, -1.102673661357727e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
